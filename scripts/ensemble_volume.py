#!/usr/bin/env python3
"""Ensemble the CV-fold encoders over one subject's volume and write the theta-map.

Each fold's encoder is swept over the volume and its output projected onto S^1,
giving one unit vector per voxel per fold.  The folds are combined by *circular*
mean -- the unit vectors are averaged and the angle of the mean is taken -- so
that folds landing at 5 deg and 355 deg agree on 0 deg rather than meeting at
180 deg, which a plain mean of the angles would give.

Two volumes come out: the continuous ensemble theta, and its quantisation,
each voxel labelled by the anchor its ensembled direction is closest to. Both
are rotated so the neutral class sits at zero -- theta is measured from the
neutral anchor and wrapped to [-pi, pi), and the labels are 0 neutral,
1 positive, 2 negative -- so that the background of either volume reads as
neutral. The trained models are untouched; this is relabelling on the way out.

With `--mask`, both volumes are zeroed outside the mask (any nonzero voxel of
a binary or multi-label mask counts as inside), i.e. set to neutral.

    python scripts/ensemble_volume.py --pha sub-001_pha.nii.gz \
        --mag sub-001_mag.nii.gz --checkpoints runs/cv5/fold*/model.pt \
        --out-dir ./results/sub-001 [--mask sub-001_lesions.nii.gz]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contrastive_prl_detection import contrastive as ct
from contrastive_prl_detection.dataset import load_norm, load_ras
from contrastive_prl_detection.inference import load_model, model_sweep, save_nifti
from contrastive_prl_detection.net import receptive_field_loss

#: Output label for each anchor index (positive, neutral, negative) of `ct`.
SEG_LABELS = np.array([1, 0, 2], dtype=np.int16)
SEG_NAMES = ("neutral (~)", "positive (+)", "negative (-)")   # indexed by label


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pha", type=Path, required=True, help="phase volume")
    p.add_argument("--mag", type=Path, required=True, help="magnitude volume")
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                   help="the fold checkpoints to ensemble, e.g. runs/cv5/fold*/model.pt")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mask", type=Path, default=None,
                   help="optional label mask; nonzero voxels are kept, the rest "
                        "of theta and seg are set to 0 (neutral)")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tile", type=int, default=64)
    p.add_argument("--halo", type=int, default=None,
                   help="default: the voxels per axis the encoder eats, read off "
                        "the checkpoint's own width (`tiled` tiles exactly)")
    return p.parse_args(argv)


@torch.inference_mode()
def ensemble_z(checkpoints, mag, pha, device, tile=64, halo=None):
    """Mean S^1 direction over the folds, plus the loss geometry they share.

    Returns the *unnormalised* mean vector: its length is the folds' resultant
    length, which `main` reports as the agreement between them.
    """
    x = torch.stack([mag, pha], dim=0).unsqueeze(0).to(device)   # (1, 2, D, H, W)
    z_sum, geom = None, None
    for ckpt_path in checkpoints:
        model, tau, anchors_deg, ckpt = load_model(ckpt_path, device)
        if geom is None:
            geom = (tau, anchors_deg)
        elif geom != (tau, anchors_deg):
            # Different anchors or temperature means the folds' angles are not
            # in the same frame, and averaging them would be meaningless.
            raise SystemExit(f"{ckpt_path}: tau/anchors {(tau, anchors_deg)} differ "
                             f"from the first checkpoint's {geom}")
        # `tiled` assumes the sweep loses exactly `halo` voxels per axis, so the
        # halo is the encoder's receptive-field loss unless the caller insists.
        h = halo or receptive_field_loss(ckpt["arch"]["w"])
        z = ct.project(model_sweep(model, x, tile, h), dim=1)      # (1, 2, D, H, W)
        z_sum = z if z_sum is None else z_sum + z
        del model
        print(f"  swept {ckpt_path}")
    return (z_sum / len(checkpoints)).squeeze(0), *geom


def main(argv=None):
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    mag, pha = load_norm(args.mag), load_norm(args.pha)
    mask = None
    if args.mask is not None:
        mask = load_ras(args.mask).numpy() != 0
        if mask.shape != tuple(mag.shape):
            raise SystemExit(f"{args.mask}: shape {mask.shape} does not match "
                             f"the volume's {tuple(mag.shape)}")
    print(f"ensembling {len(args.checkpoints)} folds over {tuple(mag.shape)}:")
    z_bar, tau, anchors_deg = ensemble_z(args.checkpoints, mag, pha, device,
                                         tile=args.tile, halo=args.halo)

    # The mean direction, renormalised onto S^1 before it is scored: only its
    # angle is the ensemble's answer, its length is the folds' agreement.
    z = ct.project(z_bar, dim=0)
    # Measured from the neutral anchor and wrapped to [-pi, pi), so neutral is
    # 0 and the seam falls between positive and negative, away from neutral.
    neutral = np.radians(anchors_deg[1])
    theta = ct.theta(z, dim=0).cpu().numpy() - neutral
    theta = (np.remainder(theta + np.pi, 2 * np.pi) - np.pi).astype(np.float32)
    logits = ct.logits(z, ct.make_anchors(device, anchors_deg=anchors_deg),
                       tau=tau, dim=0)                            # (3, D, H, W)
    seg = SEG_LABELS[logits.argmax(0).cpu().numpy()]
    if mask is not None:
        theta[~mask] = 0
        seg[~mask] = 0

    save_nifti(theta, args.mag, args.out_dir / "theta_ensemble.nii.gz")
    save_nifti(seg, args.mag, args.out_dir / "seg_ensemble.nii.gz", dtype=np.int16)

    # Stats over the mask when there is one, else over the whole volume.
    keep = np.ones(seg.shape, bool) if mask is None else mask
    counts = np.bincount(seg[keep], minlength=3)
    anchors_out = [np.degrees(np.angle(np.exp(1j * (np.radians(a) - neutral))))
                   for a in anchors_deg]
    print("theta anchors after rotation (deg): "
          + ", ".join(f"{n.split()[0]} {a:+.0f}"
                      for n, a in zip(ct.CLASS_NAMES, anchors_out)))
    print(f"theta range: [{theta[keep].min():.4f}, {theta[keep].max():.4f}] rad")
    print(f"mean fold agreement (resultant length): "
          f"{z_bar.norm(dim=0).cpu().numpy()[keep].mean():.4f}")
    where = "whole volume" if mask is None else "inside mask"
    print(f"voxel labels ({keep.sum()} voxels, {where}):")
    for label, (name, c) in enumerate(zip(SEG_NAMES, counts)):
        print(f"  {label} {name:>14}: {c:>10d} voxels ({100 * c / keep.sum():5.2f}%)")
    print(f"wrote theta_ensemble.nii.gz, seg_ensemble.nii.gz to {args.out_dir}")


if __name__ == "__main__":
    main()
