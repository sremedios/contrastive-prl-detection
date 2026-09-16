#!/usr/bin/env python3
"""Ensemble the CV-fold encoders over one subject's volume and write the theta-map.

Each fold's encoder is swept over the volume and its output projected onto S^1,
giving one unit vector per voxel per fold.  The folds are combined by *circular*
mean -- the unit vectors are averaged and the angle of the mean is taken -- so
that folds landing at 5 deg and 355 deg agree on 0 deg rather than meeting at
180 deg, which a plain mean of the angles would give.

Two volumes come out: the continuous ensemble theta in [0, 2pi), and its
quantisation, each voxel labelled by the anchor its ensembled direction is
closest to (0 positive, 1 neutral, 2 negative).

    python scripts/ensemble_volume.py --pha sub-001_pha.nii.gz \
        --mag sub-001_mag.nii.gz --checkpoints runs/cv5/fold*/model.pt \
        --out-dir ./results/sub-001
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contrastive_prl_detection import contrastive as ct
from contrastive_prl_detection.dataset import load_norm
from contrastive_prl_detection.inference import load_model, model_sweep, save_nifti
from contrastive_prl_detection.net import receptive_field_loss


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pha", type=Path, required=True, help="phase volume")
    p.add_argument("--mag", type=Path, required=True, help="magnitude volume")
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                   help="the fold checkpoints to ensemble, e.g. runs/cv5/fold*/model.pt")
    p.add_argument("--out-dir", type=Path, required=True)
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
    print(f"ensembling {len(args.checkpoints)} folds over {tuple(mag.shape)}:")
    z_bar, tau, anchors_deg = ensemble_z(args.checkpoints, mag, pha, device,
                                         tile=args.tile, halo=args.halo)

    # The mean direction, renormalised onto S^1 before it is scored: only its
    # angle is the ensemble's answer, its length is the folds' agreement.
    z = ct.project(z_bar, dim=0)
    theta = np.remainder(ct.theta(z, dim=0).cpu().numpy(), 2 * np.pi)
    logits = ct.logits(z, ct.make_anchors(device, anchors_deg=anchors_deg),
                       tau=tau, dim=0)                            # (3, D, H, W)
    seg = logits.argmax(0).cpu().numpy().astype(np.int16)

    save_nifti(theta, args.mag, args.out_dir / "theta_ensemble.nii.gz")
    save_nifti(seg, args.mag, args.out_dir / "seg_ensemble.nii.gz", dtype=np.int16)

    counts = np.bincount(seg.ravel(), minlength=3)
    print(f"theta range: [{theta.min():.4f}, {theta.max():.4f}] rad")
    print(f"mean fold agreement (resultant length): "
          f"{z_bar.norm(dim=0).mean().item():.4f}")
    print(f"voxel labels ({seg.size} voxels):")
    for name, c in zip(ct.CLASS_NAMES, counts):
        print(f"  {name:>14}: {c:>10d} voxels ({100 * c / seg.size:5.2f}%)")
    print(f"wrote theta_ensemble.nii.gz, seg_ensemble.nii.gz to {args.out_dir}")


if __name__ == "__main__":
    main()
