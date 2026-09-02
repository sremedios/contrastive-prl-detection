"""Weights & Biases logging: loss curves plus a periodic theta-map of a volume.

`wandb` is imported lazily, so nothing here is a hard dependency: with
`--no-wandb`, or with the package absent, `Logger` is inert and training runs
unchanged.

The visual progress check mirrors the notebook's validation cell rather than the
patch scatter alone: a withheld volume is swept end to end and its theta-map is
rendered at fixed slices, so successive steps are directly comparable.
"""

import torch

from . import contrastive as ct
from .dataset import get_fpaths, load_ras
from .inference import infer_volume, pick_slices, wrap_theta


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
                 dpi=200):
        self.subj_id, self.is_pos = subj_id, is_pos
        self.cohort = "pos" if is_pos else "neg"
        #: W&B key prefix. Includes the cohort so a positive and a negative
        #: subject sharing an id do not collide into one panel.
        self.tag = f"volume/{self.cohort}/{subj_id}"
        self.tile, self.halo, self.dpi = tile, halo, dpi

        pha_fpath, mag_fpath, prl_fpath, _, _ = get_fpaths(root, subj_id, pos=is_pos)
        self.pha = load_ras(pha_fpath)
        self.mag = load_ras(mag_fpath)
        self.prl = load_ras(prl_fpath).numpy() if is_pos else None
        self.slices = pick_slices(self.prl, self.mag, n_slices, is_pos)

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


def make_probes(pos_root, neg_root, withheld_ids, n_slices=4, tile=64, halo=32,
                dpi=200):
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
                                          halo, dpi))
            except Exception as e:                     # a probe is never worth a crash
                print(f"skipping volume probe for {subj_id}: {type(e).__name__}: {e}")
    return probes
