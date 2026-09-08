#!/usr/bin/env python3
"""K-fold cross-validation over subjects: one `scripts/train.py` run per fold.

`train.py` already knows how to hold subjects out of the patch pool
(`--withheld-ids`), so this script only decides *which* subjects each fold
withholds, and then runs training once per fold in a fresh process -- nothing
leaks between folds, neither CUDA memory nor the W&B run.

The split is over subjects, not patches, and is dealt per cohort: the positive
and negative subject lists are shuffled independently and handed round-robin to
`--folds` groups, so every fold withholds roughly the same number of each cohort
even when the cohorts differ in size. `--split-seed` fixes the deal, so the same
seed over the same data directory always reproduces the same folds -- which is
what makes a fold runnable on its own, days apart, and still comparable.

Everything after `--` is passed through to `train.py` unchanged.

    # all five folds, back to back
    python scripts/cross_validate.py --data-dir ./data --out-dir runs/cv5 -- \
        --device cuda:0 --n-patches 500000 --wandb

    # one fold only, e.g. as a single task of a cluster array job
    python scripts/cross_validate.py --data-dir ./data --out-dir runs/cv5 \
        --fold 2 -- --device cuda:0 --wandb
"""

import argparse
import json
import random
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contrastive_prl_detection.dataset import subject_ids_in

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"

# Flags this script derives from the fold. Accepting them in the pass-through
# would silently override the split and make the folds incomparable.
RESERVED = {"--data-dir", "--out", "--withheld-ids", "--withhold-index",
            "--no-withhold", "--wandb-name", "--val-plot-dir"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("./data"),
                   help="directory holding {pos,neu,neg}_patches")
    p.add_argument("--out-dir", type=Path, default=Path("./cv_runs"),
                   help="one fold{k}/ subdirectory per fold, plus the manifests")
    p.add_argument("--folds", type=int, default=5,
                   help="number of folds the subjects are dealt into")
    p.add_argument("--fold", type=int, nargs="*", default=None,
                   help="which folds to actually run (default: all of them). "
                        "The split does not depend on this, so folds can be run "
                        "one at a time or in parallel on separate GPUs")
    p.add_argument("--split-seed", type=int, default=0,
                   help="seed for the subject shuffle; fixes the fold assignment")
    p.add_argument("--wandb-name-prefix", default=None,
                   help="W&B run name per fold, as <prefix>-fold<k>")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip a fold whose model.pt is already there (resume a sweep)")
    p.add_argument("--keep-going", action="store_true",
                   help="carry on with the remaining folds if one fails")
    p.add_argument("--dry-run", action="store_true",
                   help="print the split and the train.py commands, run nothing")
    p.add_argument("train_args", nargs=argparse.REMAINDER,
                   help="arguments after `--` are passed to train.py verbatim")
    return p.parse_args(argv)


def make_folds(dirs, n_folds, seed):
    """Deal each cohort's subjects round-robin into `n_folds` withheld groups.

    Per cohort rather than over the pooled ids: pooling would let one fold
    withhold only positives, and its validation accuracy would then measure
    something different from its neighbours'.
    """
    folds = [[] for _ in range(n_folds)]
    for name in ("pos", "neg"):
        ids = subject_ids_in(dirs[name])
        if not ids:
            raise SystemExit(f"no {name} patches in {dirs[name]}; "
                             "run prepare_data.py first")
        if len(ids) < n_folds:
            print(f"warning: {len(ids)} {name} subjects for {n_folds} folds -- "
                  f"{n_folds - len(ids)} fold(s) withhold no {name} subject")
        shuffled = list(ids)
        random.Random(seed).shuffle(shuffled)
        for i, sid in enumerate(shuffled):
            folds[i % n_folds].append(sid)
    return [sorted(f) for f in folds]


def check_id_collisions(dirs, folds):
    """Refuse to run if one subject id is a substring of another.

    `TrainSet._collect` decides membership with `w in f.name`, so withholding
    `sub-01` would also hold out `sub-011` -- that subject would leave the
    training pool of one fold and appear in its validation pool as well, and the
    folds would quietly stop being disjoint. Cheap to check, invisible if missed.
    """
    ids = set()
    for d in dirs.values():
        ids |= set(subject_ids_in(d))
    withheld = {sid for fold in folds for sid in fold}
    bad = sorted((a, b) for a in withheld for b in ids
                 if a != b and a in b)
    if bad:
        pairs = "; ".join(f"{a!r} is inside {b!r}" for a, b in bad)
        raise SystemExit(
            f"subject ids overlap as substrings ({pairs}). Patch membership is "
            "matched by substring, so these folds would not be disjoint. Rename "
            "the patches, or make TrainSet match on the parsed subject id.")


def build_cmd(args, k, withheld, passthrough):
    fold_dir = args.out_dir / f"fold{k}"
    cmd = [sys.executable, str(TRAIN_SCRIPT),
           "--data-dir", str(args.data_dir),
           "--out", str(fold_dir / "model.pt"),
           "--val-plot-dir", str(fold_dir / "plots")]
    if args.wandb_name_prefix:
        cmd += ["--wandb-name", f"{args.wandb_name_prefix}-fold{k}"]
    # Last of ours, so the pass-through's own flags terminate the nargs="*".
    cmd += ["--withheld-ids", *withheld]
    return cmd + passthrough


def parse_summary(lines):
    """The last JSON object train.py printed, or None if it never got that far."""
    starts = [i for i, line in enumerate(lines) if line.rstrip("\n") == "{"]
    for i in reversed(starts):
        try:
            return json.loads("".join(lines[i:]))
        except json.JSONDecodeError:
            continue
    return None


def run_fold(cmd):
    """Run one training process, echoing its stdout as it goes.

    Only stdout is captured, so it can be scanned for train.py's closing JSON;
    stderr stays attached to this terminal and the tqdm bar renders live.
    """
    lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    return proc.wait(), parse_summary(lines)


def report(results):
    """Per-fold table plus the mean that is the point of running CV at all."""
    print("\n===== cross-validation summary =====")
    for r in results:
        acc = r.get("val_accuracy")
        acc = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        print(f"fold {r['fold']}: val accuracy {acc}  "
              f"[{', '.join(r['withheld_ids'])}]"
              + ("" if r["returncode"] == 0 else f"  FAILED (rc {r['returncode']})"))
    accs = [r["val_accuracy"] for r in results
            if r["returncode"] == 0 and isinstance(r.get("val_accuracy"), float)]
    if accs:
        spread = f" +/- {statistics.stdev(accs):.4f}" if len(accs) > 1 else ""
        print(f"mean val accuracy {statistics.mean(accs):.4f}{spread} "
              f"over {len(accs)} fold(s)")


def main(argv=None):
    args = parse_args(argv)

    passthrough = list(args.train_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    clash = sorted(RESERVED.intersection(a.split("=")[0] for a in passthrough))
    if clash:
        raise SystemExit(f"{', '.join(clash)}: set per fold by this script; drop it "
                         "from the arguments after `--`")
    if args.folds < 2:
        raise SystemExit(f"--folds {args.folds}: need at least 2 folds")

    dirs = {k: args.data_dir.resolve() / f"{k}_patches" for k in ("pos", "neu", "neg")}
    folds = make_folds(dirs, args.folds, args.split_seed)
    check_id_collisions(dirs, folds)

    which = sorted(set(args.fold)) if args.fold else list(range(args.folds))
    out_of_range = [k for k in which if not 0 <= k < args.folds]
    if out_of_range:
        raise SystemExit(f"--fold {out_of_range}: outside 0..{args.folds - 1}")

    print(f"{args.folds}-fold subject split (seed {args.split_seed}):")
    for k, withheld in enumerate(folds):
        print(f"  {'*' if k in which else ' '} fold {k}: "
              f"withhold {', '.join(withheld) or '(none)'}")

    if args.dry_run:
        print("\ncommands:")
        for k in which:
            print("  " + " ".join(shlex.quote(c)
                                  for c in build_cmd(args, k, folds[k], passthrough)))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "folds.json").write_text(json.dumps(
        {"folds": args.folds, "split_seed": args.split_seed,
         "data_dir": str(args.data_dir.resolve()),
         "withheld": {str(k): w for k, w in enumerate(folds)}}, indent=2))

    results = []
    for k in which:
        withheld = folds[k]
        fold_dir = args.out_dir / f"fold{k}"
        if not withheld:
            print(f"\n=== fold {k}: no subjects to withhold, skipping ===")
            continue
        if args.skip_existing and (fold_dir / "model.pt").exists():
            print(f"\n=== fold {k}: {fold_dir / 'model.pt'} exists, skipping ===")
            continue

        fold_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_cmd(args, k, withheld, passthrough)
        print(f"\n=== fold {k}/{args.folds - 1}: withholding "
              f"{', '.join(withheld)} ===")
        print("  " + " ".join(shlex.quote(c) for c in cmd), flush=True)

        t0 = time.time()
        code, summary = run_fold(cmd)
        results.append({"fold": k, "withheld_ids": withheld,
                        "returncode": code, "seconds": round(time.time() - t0, 1),
                        "out": str((fold_dir / "model.pt").resolve()),
                        **(summary or {})})
        # Rewritten after every fold, so an interrupted sweep still leaves the
        # folds that did finish on disk.
        (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))

        if code != 0:
            print(f"fold {k} failed with exit code {code}")
            if not args.keep_going:
                report(results)
                raise SystemExit(code)

    report(results)
    print(f"\nwrote {(args.out_dir / 'results.json').resolve()}")
    if any(r["returncode"] != 0 for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
