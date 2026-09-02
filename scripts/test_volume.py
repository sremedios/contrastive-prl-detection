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
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contrastive_prl_detection import contrastive as ct
from contrastive_prl_detection.dataset import get_fpaths, load_mag, load_ras
from contrastive_prl_detection.inference import (infer_volume, load_model,
                                                 pick_slices, save_nifti,
                                                 wrap_theta)


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
    mag = load_mag(mag_fpath)
    prl = load_ras(prl_fpath) if args.is_pos else None

    theta, seg, probs = infer_volume(model, mag, pha, device, tau, anchors_deg,
                                     tile=args.tile, halo=args.halo)

    # theta wrapped into [0, 2pi) so it lines up with the cyclic colormap's range
    y_hat = wrap_theta(theta)
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
