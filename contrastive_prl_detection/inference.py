"""Whole-volume inference: the tiled sweep and its decoded outputs.

Shared by `scripts/test_volume.py` and by the training-time W&B probe, so both
go through exactly the same geometry.
"""

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from . import contrastive as ct
from .dataset import unload_ras
from .net import resnet3d, tiled


def load_model(ckpt_path, device):
    """Rebuild a model from a checkpoint, returning it with its loss geometry."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = resnet3d(**ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tau = ckpt.get("tau", ct.TAU)
    anchors_deg = tuple(ckpt.get("anchors_deg", ct.ANCHORS_DEG))
    return model, tau, anchors_deg, ckpt


def model_sweep(model, x, tile=64, halo=32):
    """Tiled forward pass, with the border the encoder eats restored by padding.

    Verified voxel-exact against the patch classifier: the value at `[..., i, j, k]`
    equals a direct forward pass of the patch centred on `(i, j, k)`.
    """
    if min(x.shape[-3:]) <= halo:
        raise RuntimeError(f"volume {tuple(x.shape[-3:])} too small for halo={halo}")
    out = tiled(model, x, tile=tile, halo=halo)
    pad = ((x.shape[-1] - out.shape[-1]) // 2,
           (x.shape[-2] - out.shape[-2]) // 2,
           (x.shape[-3] - out.shape[-3]) // 2)
    if any(p < 0 for p in pad):
        raise RuntimeError(f"tiled output {tuple(out.shape)} larger than input "
                           f"{tuple(x.shape)}; check tile/halo")
    return F.pad(out, (pad[0], pad[0], pad[1], pad[1], pad[2], pad[2]), mode="replicate")


@torch.inference_mode()
def infer_volume(model, mag, pha, device, tau=ct.TAU, anchors_deg=ct.ANCHORS_DEG,
                 tile=64, halo=32):
    """Returns (theta, seg, probs), each shaped like the input volume."""
    was_training = model.training
    model.eval()
    try:
        val_inp = torch.stack([mag, pha], dim=0).unsqueeze(0)       # (1, 2, D, H, W)
        u = model_sweep(model, val_inp.to(device), tile, halo)      # (1, 2, D, H, W)

        z = ct.project(u, dim=1)
        logits = ct.logits(z, ct.make_anchors(device, anchors_deg=anchors_deg),
                           tau=tau, dim=1)                          # (1, 3, D, H, W)
        seg = logits.argmax(1)                                      # (1, D, H, W)
        probs = logits.softmax(1)                                   # (1, 3, D, H, W)
        theta = ct.theta(z, dim=1)                                  # (1, D, H, W)
        return theta.squeeze(0), seg.squeeze(0), probs.squeeze(0)
    finally:
        model.train(was_training)


def wrap_theta(theta):
    """Theta into [0, 2pi) as a numpy array, matching the cyclic colormap's range."""
    t = torch.remainder(theta, 2 * torch.pi) if torch.is_tensor(theta) \
        else np.remainder(theta, 2 * np.pi)
    return t.detach().cpu().numpy() if torch.is_tensor(t) else t


def pick_slices(lesions, mag, n):
    """Slice indices for the figure: spread over `lesions` when there are any.

    Whichever lesion mask the caller cares about -- the rim segmentation for a
    positive subject, `reg_separation` for a negative one -- so that a figure
    masked to that mask lands on slices where it has something to show. Falls
    back to a spread through the middle of the volume when the mask is empty.
    """
    if lesions is not None and (np.asarray(lesions) > 0.5).any():
        labels, _ = ndimage.label(np.asarray(lesions) > 0.5,
                                  structure=ndimage.generate_binary_structure(3, 3))
        starts = sorted({s[2].start for s in ndimage.find_objects(labels)})
        if starts:
            idx = np.linspace(0, len(starts) - 1, min(n, len(starts))).round().astype(int)
            return [starts[i] for i in idx]
    depth = mag.shape[-1]
    return [int(v) for v in np.linspace(0.3, 0.7, n) * depth]


def save_nifti(arr, ref_fpath, out_fpath, dtype=np.float32):
    """Write `arr` (in load_ras orientation) back in the reference's orientation."""
    ref = nib.load(ref_fpath)
    img = nib.Nifti1Image(unload_ras(arr).astype(dtype), ref.affine, ref.header)
    img.set_data_dtype(dtype)
    nib.save(img, out_fpath)
    return out_fpath
