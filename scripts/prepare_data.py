#!/usr/bin/env python3
"""Extract training patches from the PRL dataset and save them to disk.

Positives come from the rim-lesion segmentation of the PRL-positive cohort (one
patch per connected component).  Negatives and neutrals come from the
PRL-negative cohort: negatives are centred on lesion voxels of the `reg_separation`
mask, neutrals on voxels whose whole patch is lesion-free and whose centre is inside
the brain mask.

Each patch is saved as a `(2, *patch_size)` float32 tensor `[magnitude, phase]`
named `{subject_id}-{kind}_patch_{index:04d}.pt`, so the split by subject can be
recovered from the filename at training time.

Run once, e.g.

    python scripts/prepare_data.py \
        --pos-root /iacl/pg25/jinwei/PRL_dataset/PRL_pos \
        --neg-root /iacl/pg25/jinwei/PRL_dataset/PRL_neg \
        --out-dir ./data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contrastive_prl_detection.dataset import (get_fpaths, list_subject_ids,
                                               load_norm, load_ras,
                                               mismatched_shapes,
                                               missing_roles, patch_fname)
from contrastive_prl_detection.patch_utils import (get_neg_patches,
                                                   get_neu_patches,
                                                   get_pos_patches)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pos-root", type=Path, required=True,
                   help="root of the PRL-positive cohort (one directory per subject)")
    p.add_argument("--neg-root", type=Path, required=True,
                   help="root of the PRL-negative cohort (one directory per subject)")
    p.add_argument("--out-dir", type=Path, default=Path("./data"),
                   help="patches are written to {out-dir}/{pos,neu,neg}_patches")
    p.add_argument("--patch-size", type=int, nargs=3, default=(33, 33, 33),
                   metavar=("D", "H", "W"))
    p.add_argument("--n-neg", type=int, default=50,
                   help="negative patches sampled per negative subject")
    p.add_argument("--n-neu", type=int, default=50,
                   help="neutral patches sampled per negative subject")
    p.add_argument("--exclude-ids", nargs="*", default=[],
                   help="subject ids to skip entirely")
    p.add_argument("--on-incomplete", choices=("skip", "fail"), default="skip",
                   help="what to do with an unusable subject -- one missing an "
                        "expected volume, or one whose volumes disagree on shape "
                        "(default: report it and carry on)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true",
                   help="re-extract subjects that already have patches on disk")
    return p.parse_args(argv)


def subject_problem(root, subj_id, pos):
    """Why `subj_id` cannot be used, as one line, or None when it is fine."""
    missing = missing_roles(root, subj_id, pos=pos)
    if missing:
        return "no " + ", ".join(f"{r} (*{pat}*)" for r, pat in missing.items())
    bad = mismatched_shapes(root, subj_id, pos=pos)
    if bad:
        return "volumes on different grids: " + ", ".join(
            f"{r} {tuple(shape)} ({fpath.name})" for r, (fpath, shape) in bad.items())
    return None


def preflight(root, subject_ids, pos, args):
    """Split subjects into usable and unusable before any volume is loaded.

    Two ways a subject used to break the run minutes in: a missing volume, which
    surfaced as an IndexError, and volumes on different grids, which surfaced as
    a broadcast RuntimeError inside patch sampling. Both are decided up front,
    from the file listing and the NIfTI headers, so no voxel data is read here.
    """
    ok, bad = [], {}
    for subj_id in subject_ids:
        problem = subject_problem(root, subj_id, pos)
        (ok.append(subj_id) if problem is None else bad.update({subj_id: problem}))

    if bad:
        label = "positive" if pos else "negative"
        print(f"\n{len(bad)} of {len(subject_ids)} {label} subjects are unusable:")
        for subj_id, problem in bad.items():
            print(f"  {subj_id}: {problem}")
        if args.on_incomplete == "fail":
            raise SystemExit("aborting (--on-incomplete fail). Pass --exclude-ids "
                             "to drop them, or --on-incomplete skip to continue.")
        print(f"  -> skipping them; {len(ok)} {label} subjects will be processed.\n")
    return ok, bad


def report_geometry(extents):
    """Print what the lesion annotations and the phase range actually look like.

    The bounding-box extents say which axis carries the 2D annotation: the one
    with extent 1 throughout.
    """
    if not extents:
        return
    arr = np.asarray(extents)
    print("\npositive component bounding-box extents, per axis "
          f"({len(arr)} components):")
    for ax in range(arr.shape[1]):
        col = arr[:, ax]
        flag = "  <- 2D annotation axis" if (col == 1).all() else ""
        print(f"  axis {ax}: median {np.median(col):5.1f}  min {col.min():3d}  "
              f"max {col.max():3d}{flag}")
    if not (arr == 1).all(axis=0).any():
        print("  note: no axis is uniformly 1, so the annotation is not purely 2D")


def save_patches(xs, out_dir, subj_id, kind):
    """Write each patch of a stacked `(N, 2, *patch_size)` tensor to its own file.

    `.clone()`, not `.contiguous()`: `xs[i]` is a view into the stacked tensor
    and a slice along dim 0 is already contiguous, so `.contiguous()` returns
    the same object and `torch.save` writes the whole underlying storage -- all
    N patches -- into every file. That inflates the set by a factor of N and
    makes each `torch.load` at training time read N patches to use one.
    """
    for i, x in enumerate(xs):
        torch.save(x.clone(), out_dir / patch_fname(subj_id, kind, i))
    return len(xs)


def already_done(out_dir, subj_id, kind):
    return any(out_dir.glob(f"{subj_id}-{kind}_patch_*.pt"))


def prepare_positive(args, dirs, rng):
    subject_ids = [s for s in list_subject_ids(args.pos_root)
                   if s not in args.exclude_ids]
    subject_ids, skipped = preflight(args.pos_root, subject_ids, True, args)
    total = 0
    extents = []
    for subj_id in tqdm(subject_ids, desc="pos subjects"):
        if not args.overwrite and already_done(dirs["pos"], subj_id, "pos"):
            continue
        pha_fpath, mag_fpath, prl_fpath, _, _ = get_fpaths(args.pos_root, subj_id, pos=True)
        pha = load_norm(pha_fpath)
        mag = load_norm(mag_fpath)
        prl = load_ras(prl_fpath)

        xs_pos, ext = get_pos_patches(mag, pha, prl, args.patch_size,
                                      return_extents=True)
        extents += ext
        total += save_patches(xs_pos, dirs["pos"], subj_id, "pos")
    report_geometry(extents)
    return len(subject_ids), total, skipped


def prepare_negative(args, dirs, rng):
    subject_ids = [s for s in list_subject_ids(args.neg_root)
                   if s not in args.exclude_ids]
    subject_ids, skipped = preflight(args.neg_root, subject_ids, False, args)
    n_neg = n_neu = 0
    for subj_id in tqdm(subject_ids, desc="neg subjects"):
        done = (already_done(dirs["neg"], subj_id, "neg")
                and already_done(dirs["neu"], subj_id, "neu"))
        if not args.overwrite and done:
            continue
        pha_fpath, mag_fpath, _, brainmask_fpath, aultra_fpath = get_fpaths(
            args.neg_root, subj_id, pos=False)
        pha = load_norm(pha_fpath)
        mag = load_norm(mag_fpath)
        seg = load_ras(aultra_fpath)
        seg[seg != 0] = 1
        brainmask = load_ras(brainmask_fpath)

        xs_neg = get_neg_patches(mag, pha, seg, args.patch_size, n=args.n_neg, rng=rng)
        xs_neu = get_neu_patches(mag, pha, seg, brainmask, args.patch_size,
                                 n=args.n_neu, rng=rng)

        n_neg += save_patches(xs_neg, dirs["neg"], subj_id, "neg")
        n_neu += save_patches(xs_neu, dirs["neu"], subj_id, "neu")
    return len(subject_ids), n_neg, n_neu, skipped


def main(argv=None):
    args = parse_args(argv)
    args.patch_size = tuple(args.patch_size)
    rng = np.random.default_rng(args.seed)

    dirs = {k: args.out_dir.resolve() / f"{k}_patches" for k in ("pos", "neu", "neg")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    n_pos_subj, n_pos, skip_pos = prepare_positive(args, dirs, rng)
    n_neg_subj, n_neg, n_neu, skip_neg = prepare_negative(args, dirs, rng)

    print(f"\npatch size {args.patch_size}, written under {args.out_dir.resolve()}")
    print(f"  positive: {n_pos:6d} new patches from {n_pos_subj} subjects -> {dirs['pos']}")
    print(f"  neutral : {n_neu:6d} new patches from {n_neg_subj} subjects -> {dirs['neu']}")
    print(f"  negative: {n_neg:6d} new patches from {n_neg_subj} subjects -> {dirs['neg']}")
    for k, d in dirs.items():
        print(f"  {k} total on disk: {len(list(d.glob('*.pt')))}")

    skipped = {**skip_pos, **skip_neg}
    if skipped:
        print(f"\nskipped {len(skipped)} unusable subject(s): "
              f"{', '.join(sorted(skipped))}")


if __name__ == "__main__":
    main()
