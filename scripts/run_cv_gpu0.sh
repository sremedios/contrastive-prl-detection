#!/usr/bin/env bash
# Folds 0, 1, 2 of the 5-fold CV, in series on cuda:0.
# Pair with run_cv_gpu1.sh (folds 3, 4 on cuda:1) in another tmux pane.
# Run from the repo root. When both panes are done:
#     python scripts/cross_validate.py --out-dir runs/cv5 --summarize

for k in 0 1 2; do
    python scripts/cross_validate.py \
        --data-dir /tmp/PRL_PATCHES \
        --out-dir runs/cv5 \
        --fold $k \
        --wandb-name-prefix cv5 \
        -- \
        --device cuda:0 \
        --n-patches 1500000 \
        --wandb
done
