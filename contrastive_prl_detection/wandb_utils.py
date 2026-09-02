"""Weights & Biases logging: loss curves plus a periodic theta-map of a volume.

`wandb` is imported lazily, so nothing here is a hard dependency: with
`--no-wandb`, or with the package absent, `Logger` is inert and training runs
unchanged.

The visual progress check mirrors the notebook's validation cell rather than the
patch scatter alone: a withheld volume is swept end to end and its theta-map is
rendered at fixed slices, so successive steps are directly comparable.
"""

import torch

import numpy as np

from . import contrastive as ct
from .dataset import get_fpaths, load_norm, load_ras
from .inference import infer_volume, pick_slices, wrap_theta
from .patch_utils import get_neg_patches, get_neu_patches, get_pos_patches


class Logger:
    """Thin wrapper so callers never branch on whether W&B is enabled."""

    def __init__(self, enabled=False, project=None, entity=None, name=None,
                 mode="online", config=None):
        self.run = None
        self.enabled = False
        if not enabled:
            return
        try:
            import wandb
        except ImportError:
            print("wandb not installed (`pip install wandb`); continuing without it")
            return
        self._wandb = wandb
        self.run = wandb.init(project=project, entity=entity, name=name,
                              mode=mode, config=config or {})
        self.enabled = True
        print(f"logging to W&B: {self.run.url}")

    def log(self, data, step=None, commit=True):
        if self.enabled:
            self.run.log(data, step=step, commit=commit)

    def image(self, fig, caption=None):
        """Wrap a matplotlib figure; returns None when logging is off."""
        return self._wandb.Image(fig, caption=caption) if self.enabled else None

    def summary(self, data):
        if self.enabled:
            self.run.summary.update(data)

    def finish(self):
        if self.enabled:
            self.run.finish()


class VolumeProbe:
    """One withheld volume, held in memory, rendered as a theta-map on demand.

    The volume and its slice indices are chosen once at construction so every
    logged frame shows the same anatomy and the panels can be flipped through
    across steps like a time-lapse.
    """

    def __init__(self, root, subj_id, is_pos=True, n_slices=4, tile=64, halo=32,
                 dpi=200, n_patches=64, patch_size=(33, 33, 33), seed=0):
        self.subj_id, self.is_pos = subj_id, is_pos
        self.cohort = "pos" if is_pos else "neg"
        #: W&B key prefix. Includes the cohort so a positive and a negative
        #: subject sharing an id do not collide into one panel.
        self.tag = f"volume/{self.cohort}/{subj_id}"
        self.tile, self.halo, self.dpi = tile, halo, dpi

        pha_fpath, mag_fpath, prl_fpath, brainmask_fpath, aultra_fpath = get_fpaths(
            root, subj_id, pos=is_pos)
        self.pha = load_norm(pha_fpath)
        self.mag = load_norm(mag_fpath)
        self.prl = load_ras(prl_fpath).numpy() if is_pos else None
        self.slices = pick_slices(self.prl, self.mag, n_slices, is_pos)
        self.patches = self._sample_patches(brainmask_fpath, aultra_fpath,
                                            n_patches, patch_size, seed)

    def _sample_patches(self, brainmask_fpath, aultra_fpath, n, patch_size, seed):
        """Cut a fixed patch sample from this volume, by class index.

        Sampled once, with the same rules `prepare_data` uses for this cohort,
        so the scatter moves because the model moved and not because the sample
        was redrawn. Held out of training, and drawn live from the volume rather
        than from the handful of patches prep wrote, so the neutral and negative
        classes are far better covered than the stored pool.
        """
        rng = np.random.default_rng(seed)
        out = {}
        if self.is_pos:
            xs = get_pos_patches(self.mag, self.pha, self.prl, patch_size)
            if len(xs):
                sel = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
                out[0] = xs[sorted(sel)]                      # 0 = positive
        else:
            seg = load_ras(aultra_fpath)
            seg[seg != 0] = 1
            brainmask = load_ras(brainmask_fpath)
            neu = get_neu_patches(self.mag, self.pha, seg, brainmask, patch_size,
                                  n=n, rng=rng)
            neg = get_neg_patches(self.mag, self.pha, seg, patch_size, n=n, rng=rng)
            if len(neu):
                out[1] = neu                                  # 1 = neutral
            if len(neg):
                out[2] = neg                                  # 2 = negative
        return out

    def render(self, model, device, tau=ct.TAU, anchors_deg=ct.ANCHORS_DEG,
               step=None):
        """Returns the theta-map figure. The caller closes it."""
        from .polar_utils import plot_theta_slices

        theta, _, _ = infer_volume(model, self.mag, self.pha, device, tau,
                                   anchors_deg, tile=self.tile, halo=self.halo)
        y_hat = wrap_theta(theta)

        title = f"{self.subj_id} ({self.cohort}, withheld)"
        if step is not None:
            title += f" - step {step}"
        fig = plot_theta_slices(y_hat, self.slices, overlay=self.prl, dpi=self.dpi,
                                title=title, show=False, close=False)

        del theta, y_hat
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return fig


@torch.no_grad()
def embed_probe_patches(probes, model, device, batch=32):
    """Run the probes' held-out patches through the model. Returns (u, z, y, theta).

    The patch-wise counterpart to the theta-map: it shows whether the three
    classes separate on the circle for volumes the model never trained on.
    """
    xs, ys = [], []
    for probe in probes:
        for cls, t in sorted(probe.patches.items()):
            xs.append(t)
            ys.append(torch.full((len(t),), cls, dtype=torch.long))
    if not xs:
        return None

    x, y = torch.cat(xs), torch.cat(ys).numpy()
    was_training = model.training
    model.eval()
    try:
        us = [model(x[i:i + batch].to(device)).flatten(1).cpu()
              for i in range(0, len(x), batch)]
    finally:
        model.train(was_training)

    u = torch.cat(us).numpy()
    z, theta = ct.theta_np(u)
    return u, z, y, theta


def make_probes(pos_root, neg_root, withheld_ids, n_slices=4, tile=64, halo=32,
                dpi=200, n_patches=64, patch_size=(33, 33, 33)):
    """Build a probe per cohort root given, for whichever withheld id it holds."""
    probes = []
    for root, is_pos in ((pos_root, True), (neg_root, False)):
        if root is None:
            continue
        for subj_id in withheld_ids:
            if not (root / subj_id).is_dir():
                continue
            try:
                probes.append(VolumeProbe(root, subj_id, is_pos, n_slices, tile,
                                          halo, dpi, n_patches, patch_size))
            except Exception as e:                     # a probe is never worth a crash
                print(f"skipping volume probe for {subj_id}: {type(e).__name__}: {e}")
    return probes
