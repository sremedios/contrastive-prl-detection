#!/usr/bin/env python3
"""Train the contrastive encoder on the patches written by `prepare_data.py`.

Each step draws a (positive, neutral, negative) triplet batch, pushes all 3B
patches through the encoder to get u in R^2, projects onto S^1, scores against
the three fixed anchors at 90/210/330 degrees, and takes cross-entropy.

Validation is the leave-subjects-out half of the same patch pool: the subjects
named by `--withheld-ids` (or picked by `--withhold-index`) are held out of
training and used to report nearest-anchor accuracy, plus the R^2 / S^1 scatter
of `polar_utils.plot_both_views`.

    python scripts/train.py --data-dir ./data --device cuda:1 --out model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contrastive_prl_detection import contrastive as ct
from contrastive_prl_detection.dataset import (TrainSet, subject_ids_in,
                                               worker_init_fn)
from contrastive_prl_detection.net import resnet3d
from contrastive_prl_detection.wandb_utils import Logger, make_probes

DEFAULT_WIDTH = (16, 32, 64, 64, 64, 64, 64, 64)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("./data"),
                   help="directory holding {pos,neu,neg}_patches")
    p.add_argument("--out", type=Path, default=Path("model.pt"))

    p.add_argument("--withheld-ids", nargs="*", default=None,
                   help="subject ids held out of training; defaults to the "
                        "--withhold-index'th positive and negative subject")
    p.add_argument("--withhold-index", type=int, default=0,
                   help="index into the sorted subject ids, for LOSO CV folds")
    p.add_argument("--no-withhold", action="store_true",
                   help="train on every subject and skip validation")

    p.add_argument("--n-patches", type=int, default=500_000,
                   help="triplets drawn per training run (one pass over the loader)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=ct.TAU)
    p.add_argument("--width", type=int, nargs="+", default=DEFAULT_WIDTH)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-augment", action="store_true",
                   help="disable the 90-degree rotations and flips on training patches")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--val-every", type=int, default=0,
                   help="run validation every N steps (0 = only at the end)")
    p.add_argument("--val-patches", type=int, default=100,
                   help="triplets per validation pass")
    p.add_argument("--val-plot-dir", type=Path, default=None,
                   help="save the R^2 / S^1 scatter here instead of showing it")
    g = p.add_argument_group("weights & biases")
    g.add_argument("--wandb", action="store_true",
                   help="log losses, validation metrics, and theta-maps to W&B "
                        "(run `wandb login` once on this machine first)")
    g.add_argument("--wandb-project", default="contrastive-prl-detection")
    g.add_argument("--wandb-entity", default=None)
    g.add_argument("--wandb-name", default=None, help="run name; default is W&B's")
    g.add_argument("--wandb-mode", default="online",
                   choices=("online", "offline", "disabled"))
    g.add_argument("--log-every", type=int, default=10,
                   help="steps between loss points")
    g.add_argument("--vol-every", type=int, default=50,
                   help="steps between whole-volume theta-map renders "
                        "(0 = only at the end). Needs --pos-root/--neg-root")
    g.add_argument("--pos-root", type=Path, default=None,
                   help="positive cohort root, so the withheld volume can be swept")
    g.add_argument("--neg-root", type=Path, default=None,
                   help="negative cohort root, likewise")
    g.add_argument("--vol-slices", type=int, default=4,
                   help="slices per rendered theta-map")
    g.add_argument("--vol-dpi", type=int, default=200,
                   help="resolution of the rendered theta-map panels")

    p.add_argument("--ckpt-every", type=int, default=0,
                   help="also write an intermediate checkpoint every N steps")
    return p.parse_args(argv)


def resolve_withheld(args, dirs):
    if args.no_withhold:
        return []
    if args.withheld_ids is not None:
        return list(args.withheld_ids)
    pos_ids = subject_ids_in(dirs["pos"])
    neg_ids = subject_ids_in(dirs["neg"])
    k = args.withhold_index
    withheld = []
    for ids, name in ((pos_ids, "positive"), (neg_ids, "negative")):
        if not ids:
            raise SystemExit(f"no {name} patches found; run prepare_data.py first")
        if k >= len(ids):
            raise SystemExit(f"--withhold-index {k} out of range for {len(ids)} "
                             f"{name} subjects")
        withheld.append(ids[k])
    return sorted(set(withheld))


def make_loader(dirs, n_patches, batch_size, withheld, invert, workers, seed,
                pin_memory=False, augment=False):
    ds = TrainSet(dirs["pos"], dirs["neu"], dirs["neg"], n_patches,
                  withheld_ids=withheld, invert=invert, seed=seed,
                  augment=augment)
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False,           # TrainSet already samples at random
                        pin_memory=pin_memory, num_workers=workers,
                        worker_init_fn=worker_init_fn if workers else None)
    return ds, loader


def check_patch_fits_encoder(model, sample, device, width):
    """The encoder must collapse a patch to 1x1x1, or `flatten(1)` is not in R^2.

    `resnet3d` loses 4 voxels per axis per block, so patch_size must be
    `4 * len(width) + 1` on every axis.
    """
    with torch.no_grad():
        out = model(sample[:1].to(device))
    if out.shape[-3:] != (1, 1, 1):
        need = 4 * len(width) + 1
        raise SystemExit(
            f"patch size {tuple(sample.shape[-3:])} does not match a {len(width)}-block "
            f"encoder: forward gives {tuple(out.shape[-3:])} instead of (1, 1, 1). "
            f"Use patches of {need}^3, or a --width with "
            f"{(sample.shape[-1] - 1) // 4} blocks.")
    return out.shape[1]


def validate(model, val_loader, device, anchors_deg, step, plot_dir,
             show_plot=False):
    u, z, y, theta = ct.embed(model, val_loader, device)
    acc, per_class = ct.accuracy_from_theta(theta, y, anchors_deg)
    model.train()

    from contrastive_prl_detection.polar_utils import plot_both_views
    savepath = None
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        savepath = plot_dir / f"embedding_step{step:07d}.png"
    fig = plot_both_views(u, z, y, theta, anchors_deg=anchors_deg,
                          show=show_plot, savepath=savepath, close=False)

    named = ", ".join(f"{n}={a:.3f}" for n, a in zip(ct.CLASS_NAMES, per_class))
    print(f"[step {step}] val accuracy {acc:.4f}  ({named})"
          + (f"  -> {savepath}" if savepath else ""))
    return acc, per_class, fig


def volume_payload(logger, probes, model, device, tau, anchors_deg, step):
    """Render each withheld volume's theta-map. Returns the images to log."""
    payload = {}
    for probe in probes:
        fig = probe.render(model, device, tau, anchors_deg, step=step)
        payload[f"{probe.tag}/theta_slices"] = logger.image(
            fig, caption=f"step {step} - slices {probe.slices}")
        plt.close(fig)
    return payload


def val_payload(logger, acc, per_class, fig, step):
    return {"val/accuracy": acc,
            **{f"val/acc_{n}": a for n, a in
               zip(("positive", "neutral", "negative"), per_class)},
            "val/embedding": logger.image(fig, caption=f"step {step}")}


def save_checkpoint(path, model, args, anchors_deg, step, val_acc):
    torch.save({
        "state_dict": model.state_dict(),
        "arch": dict(c_in=2, c_out=2, w=tuple(args.width)),
        "tau": args.tau,
        "anchors_deg": tuple(anchors_deg),
        "step": step,
        "withheld_ids": list(args.withheld_ids or []),
        "val_accuracy": val_acc,
    }, path)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dirs = {k: args.data_dir.resolve() / f"{k}_patches" for k in ("pos", "neu", "neg")}
    withheld = resolve_withheld(args, dirs)
    args.withheld_ids = withheld
    print(f"withheld subjects: {withheld or '(none)'}")

    # ===== Data =====
    pin = torch.device(args.device).type == "cuda"
    trainset, data_loader = make_loader(dirs, args.n_patches, args.batch_size,
                                        withheld, invert=False,
                                        workers=args.workers, seed=args.seed,
                                        pin_memory=pin, augment=not args.no_augment)
    print(f"train pool: {len(trainset.pos_fpaths)} pos / "
          f"{len(trainset.neu_fpaths)} neu / {len(trainset.neg_fpaths)} neg patches")

    val_loader = None
    if withheld:
        try:
            valset, val_loader = make_loader(dirs, args.val_patches,
                                             max(1, args.batch_size // 2),
                                             withheld, invert=True,
                                             workers=0, seed=args.seed + 1,
                                             pin_memory=pin)
            print(f"val pool  : {len(valset.pos_fpaths)} pos / "
                  f"{len(valset.neu_fpaths)} neu / {len(valset.neg_fpaths)} neg patches")
        except ValueError as e:
            print(f"validation disabled: {e}")

    if args.val_plot_dir is not None or args.wandb:
        matplotlib.use("Agg")

    # ===== Network =====
    device = torch.device(args.device)
    model = resnet3d(c_in=2, c_out=2, w=tuple(args.width)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    anchors = ct.make_anchors(device)

    c_out = check_patch_fits_encoder(model, trainset[0][0].unsqueeze(0), device,
                                     args.width)
    if c_out != 2:
        raise SystemExit(f"encoder must emit 2 channels for S^1, got {c_out}")

    # ===== Logging =====
    logger = Logger(enabled=args.wandb, project=args.wandb_project,
                    entity=args.wandb_entity, name=args.wandb_name,
                    mode=args.wandb_mode,
                    config={k: (str(v) if isinstance(v, Path) else v)
                            for k, v in vars(args).items()})
    probes = []
    if args.wandb and (args.pos_root or args.neg_root):
        probes = make_probes(args.pos_root, args.neg_root, withheld,
                             n_slices=args.vol_slices, dpi=args.vol_dpi)
        print(f"volume probes: {[pr.tag for pr in probes] or '(none)'}")
    elif args.wandb and args.vol_every:
        print("--vol-every needs --pos-root and/or --neg-root; skipping volume renders")

    # The theta legend never changes, so it is logged once rather than per step.
    if logger.enabled:
        from contrastive_prl_detection.polar_utils import plot_circular_colorbar
        cbar = plot_circular_colorbar(dpi=args.vol_dpi, show=False, close=False)
        logger.log({"volume/theta_colorbar": logger.image(cbar)}, step=0)
        plt.close(cbar)

    # ===== Training =====
    model.train()
    loader_pbar = tqdm(data_loader)
    step = 0
    last_loss = None
    val_acc = None

    for pos, neu, neg in loader_pbar:
        n = pos.shape[0]
        x = torch.cat([pos, neu, neg], dim=0).to(device, non_blocking=True)
        y = torch.arange(3, device=device).repeat_interleave(n)

        u = model(x).flatten(1)                            # (3n, 2)
        z = ct.project(u, dim=1)                           # onto S^1
        logits = ct.logits(z, anchors, tau=args.tau, dim=1)  # (3n, 3)

        loss = F.cross_entropy(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        step += 1
        last_loss = loss.item()
        loader_pbar.set_postfix({"loss": f"{last_loss:.4f}"})

        # Everything for this step goes in one payload: a second log() call at an
        # already-committed step makes W&B advance its own counter, which
        # desynchronises the images from the loss curve.
        payload = {}
        if args.log_every and step % args.log_every == 0:
            payload |= {"train/loss": last_loss}

        if val_loader is not None and args.val_every and step % args.val_every == 0:
            val_acc, per_class, fig = validate(model, val_loader, device,
                                               ct.ANCHORS_DEG, step, args.val_plot_dir,
                                               show_plot=args.val_plot_dir is None
                                                         and not args.wandb)
            payload |= val_payload(logger, val_acc, per_class, fig, step)
            plt.close(fig)

        if probes and args.vol_every and step % args.vol_every == 0:
            payload |= volume_payload(logger, probes, model, device, args.tau,
                                      ct.ANCHORS_DEG, step)

        if payload:
            logger.log(payload, step=step)
        if args.ckpt_every and step % args.ckpt_every == 0:
            save_checkpoint(args.out.with_suffix(f".step{step}.pt"), model, args,
                            ct.ANCHORS_DEG, step, val_acc)

    loader_pbar.close()

    # Final validation and volume render, at a step past the last training one so
    # the payload never collides with an already-committed step.
    final = {}
    if val_loader is not None and not (args.val_every and step % args.val_every == 0):
        val_acc, per_class, fig = validate(model, val_loader, device, ct.ANCHORS_DEG,
                                           step, args.val_plot_dir,
                                           show_plot=args.val_plot_dir is None
                                                     and not args.wandb)
        final |= val_payload(logger, val_acc, per_class, fig, step)
        plt.close(fig)

    # Always render the volumes once at the end, whatever --vol-every says.
    if probes and not (args.vol_every and step % args.vol_every == 0):
        final |= volume_payload(logger, probes, model, device, args.tau,
                                ct.ANCHORS_DEG, step)
    if final:
        logger.log(final, step=step + 1)

    save_checkpoint(args.out, model, args, ct.ANCHORS_DEG, step, val_acc)
    logger.summary({"final_loss": last_loss, "val_accuracy": val_acc,
                    "steps": step})
    logger.finish()
    print(f"saved {args.out.resolve()}")
    print(json.dumps({"steps": step, "final_loss": last_loss,
                      "val_accuracy": val_acc, "withheld_ids": withheld}, indent=2))


if __name__ == "__main__":
    main()
