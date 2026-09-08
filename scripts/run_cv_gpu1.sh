#!/usr/bin/env bash
# Folds 3, 4 of the 5-fold CV, in series on cuda:1.
# Pair with run_cv_gpu0.sh (folds 0, 1, 2 on cuda:0) in another tmux pane.
# Run from the repo root. When both panes are done:
#     python scripts/cross_validate.py --out-dir runs/cv5 --summarize
#
# CHECK the two cohort root paths below before launching -- if they hold none of
# the fold's withheld subjects, train.py now stops immediately and says so.

for k in 3 4; do
    python scripts/cross_validate.py \
        --data-dir /tmp/PRL_PATCHES \
        --out-dir runs/cv5 \
        --fold $k \
        --wandb-name-prefix cv5 \
        -- \
        --device cuda:1 \
        --n-patches 1500000 \
        --wandb \
        --pos-root /iacl/pg25/jinwei/PRL_dataset/PRL_pos \
        --neg-root /iacl/pg25/jinwei/PRL_dataset/PRL_neg \
        --vol-subjects 1 \
        --vol-every 1000
done
