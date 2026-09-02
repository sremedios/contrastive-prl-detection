#!/usr/bin/env python3
"""Run a trained encoder over one whole volume and write out the theta-map.

The encoder is fully convolutional, so inference is a tiled sweep (`net.tiled`)
rather than a patch loop.  Each block of `resnet3d` eats 4 voxels per axis, so
the sweep returns a volume shrunk by `4 * len(w)` voxels per axis; that border is
restored with replicate padding, giving an output the same shape as the input.

Per voxel the 2-channel output is projected onto S^1 and scored against the three
anchors, giving a ternary segmentation, class probabilities, and the continuous
theta-map that the cyclic colormap renders directly.

    python scripts/test_volume.py --checkpoint model.pt \
        --root /iacl/pg25/jinwei/PRL_dataset/PRL_pos --subject-id SUBJ001 \
        --out-dir ./results
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contrastive_prl_detection import contrastive as ct
from contrastive_prl_detection.dataset import get_fpaths, load_ras, unload_ras
from contrastive_prl_detection.net import resnet3d


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("model.pt"))
    p.add_argument("--root", type=Path, required=True,
                   help="cohort root containing the subject directory")
    p.add_argument("--subject-id", required=True)
    p.add_argument("--neg", dest="is_pos", action="store_false", default=True,
                   help="subject is from the PRL-negative cohort (no rim segmentation)")

    p.add_argument("--out-dir", type=Path, default=Path("./results"))
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tile", type=int, default=64)
    p.add_argument("--halo", type=int, default=32)
    p.add_argument("--mask", action="store_true",
                   help="zero the theta-map outside the reg_separation lesion mask")

    p.add_argument("--slices", type=int, nargs="*", default=None,
                   help="slice indices for the figure; default is spread over the "
                        "rim lesions (positive subject) or over the brain")
    p.add_argument("--n-slices", type=int, default=4)
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--show", action="store_true",
                   help="display the figure instead of only saving it")
    return p.parse_args(argv)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = resnet3d(**ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tau = ckpt.get("tau", ct.TAU)
    anchors_deg = tuple(ckpt.get("anchors_deg", ct.ANCHORS_DEG))
    return model, tau, anchors_deg, ckpt


@torch.inference_mode()
def infer_volume(model, mag, pha, device, tau, anchors_deg, tile=64, halo=32):
    """Returns (theta, seg, probs) each shaped like the input volume."""
    val_inp = torch.stack([mag, pha], dim=0).unsqueeze(0)          # (1, 2, D, H, W)
    u = model_sweep(model, val_inp.to(device), tile, halo)          # (1, 2, D, H, W)

    z = ct.project(u, dim=1)
    logits = ct.logits(z, ct.make_anchors(device, anchors_deg=anchors_deg),
                       tau=tau, dim=1)                              # (1, 3, D, H, W)
    seg = logits.argmax(1)                                          # (1, D, H, W)
    probs = logits.softmax(1)                                       # (1, 3, D, H, W)
    theta = ct.theta(z, dim=1)                                      # (1, D, H, W)
    return theta.squeeze(0), seg.squeeze(0), probs.squeeze(0)


def model_sweep(model, x, tile, halo):
    """Tiled forward pass, with the lost border restored by replicate padding."""
    out = tiled_or_whole(model, x, tile, halo)
    pad = (x.shape[-1] - out.shape[-1]) // 2, (x.shape[-2] - out.shape[-2]) // 2, \
          (x.shape[-3] - out.shape[-3]) // 2
    if any(p < 0 for p in pad):
        raise RuntimeError(f"tiled output {tuple(out.shape)} larger than input "
                           f"{tuple(x.shape)}; check --tile/--halo")
    return F.pad(out, (pad[0], pad[0], pad[1], pad[1], pad[2], pad[2]), mode="replicate")


def tiled_or_whole(model, x, tile, halo):
    from contrastive_prl_detection.net import tiled
    if min(x.shape[-3:]) <= halo:
        raise RuntimeError(f"volume {tuple(x.shape[-3:])} too small for halo={halo}")
    return tiled(model, x, tile=tile, halo=halo)


def pick_slices(prl, mag, n, is_pos):
    if is_pos and prl is not None and (prl > 0.5).any():
        labels, _ = ndimage.label(np.asarray(prl > 0.5),
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


def main(argv=None):
    args = parse_args(argv)
    if not args.show:
        matplotlib.use("Agg")
    device = torch.device(args.device)
    args.out_dir = args.out_dir.resolve() / args.subject_id
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, tau, anchors_deg, ckpt = load_model(args.checkpoint, device)
    if args.subject_id in ckpt.get("withheld_ids", []):
        print(f"note: {args.subject_id} was withheld from training in this checkpoint")

    pha_fpath, mag_fpath, prl_fpath, _, aultra_fpath = get_fpaths(
        args.root, args.subject_id, pos=args.is_pos)
    pha = load_ras(pha_fpath)
    mag = load_ras(mag_fpath)
    prl = load_ras(prl_fpath) if args.is_pos else None

    theta, seg, probs = infer_volume(model, mag, pha, device, tau, anchors_deg,
                                     tile=args.tile, halo=args.halo)

    # theta wrapped into [0, 2pi) so it lines up with the cyclic colormap's range
    y_hat = torch.remainder(theta, 2 * torch.pi).cpu().numpy()
    seg_np = seg.cpu().numpy().astype(np.int16)
    pos_prob = probs[0].cpu().numpy()

    valid = np.ones_like(seg_np, dtype=bool)
    if args.mask:
        # Out-of-mask voxels get theta 0 and label -1; class 0 is *positive*, so
        # zeroing the labels instead would read as a whole-brain detection.
        valid = (load_ras(aultra_fpath).numpy() != 0)
        y_hat = np.where(valid, y_hat, 0.0)
        seg_np = np.where(valid, seg_np, -1)
        pos_prob = np.where(valid, pos_prob, 0.0)

    save_nifti(y_hat, mag_fpath, args.out_dir / "theta.nii.gz")
    save_nifti(seg_np, mag_fpath, args.out_dir / "seg.nii.gz", dtype=np.int16)
    save_nifti(pos_prob, mag_fpath, args.out_dir / "prob_positive.nii.gz")

    inside = seg_np[valid]
    counts = np.bincount(inside.ravel(), minlength=3)
    scope = "inside the lesion mask" if args.mask else "over the whole volume"
    print(f"theta range: [{y_hat[valid].min():.4f}, {y_hat[valid].max():.4f}] rad")
    print(f"voxel labels {scope} ({inside.size} voxels):")
    for name, c in zip(ct.CLASS_NAMES, counts):
        print(f"  {name:>14}: {c:>10d} voxels ({100 * c / max(inside.size, 1):5.2f}%)")
    print(f"wrote theta.nii.gz, seg.nii.gz, prob_positive.nii.gz to {args.out_dir}")

    if not args.no_figure:
        from contrastive_prl_detection.polar_utils import plot_theta_slices
        slices = args.slices or pick_slices(
            None if prl is None else prl.numpy(), mag, args.n_slices, args.is_pos)
        overlay = prl.numpy() if args.is_pos and prl is not None else None
        fig_path = args.out_dir / "theta_slices.png"
        plot_theta_slices(y_hat, slices, overlay=overlay,
                          show=args.show, savepath=fig_path)
        print(f"wrote {fig_path} (slices {slices})")


if __name__ == "__main__":
    main()
