#!/usr/bin/env bash
# Folds 0, 1, 2 of the 5-fold CV, in series on cuda:0.
#
# Pair with run_cv_gpu1.sh, which takes folds 3 and 4 on cuda:1; launch the two
# in separate tmux panes. They share --out-dir but never the same fold, and each
# writes its own results-fold*.json, so neither clobbers the other. When both
# panes are done, merge them into one table with:
#
#     python scripts/cross_validate.py --out-dir runs/cv5 --summarize
#
# Every setting below can be overridden from the environment, e.g.
#     DEVICE=cuda:2 OUT_DIR=runs/cv5_rerun ./scripts/run_cv_gpu0.sh

set -uo pipefail
cd "$(dirname "$0")/.."          # so relative OUT_DIR lands at the repo root

DATA_DIR="${DATA_DIR:-/tmp/PRL_PATCHES}"
OUT_DIR="${OUT_DIR:-runs/cv5}"
DEVICE="${DEVICE:-cuda:0}"
FOLDS=(${FOLDS_OVERRIDE:-0 1 2})
N_PATCHES="${N_PATCHES:-1500000}"
SPLIT_SEED="${SPLIT_SEED:-0}"
NAME_PREFIX="${NAME_PREFIX:-cv5}"
WANDB="${WANDB:-1}"              # WANDB=0 for a quick run with no logging
POS_ROOT="${POS_ROOT:-}"         # set both to get the withheld-volume renders
NEG_ROOT="${NEG_ROOT:-}"

extra=()
[ "$WANDB" = "1" ] && extra+=(--wandb)
[ -n "$POS_ROOT" ] && extra+=(--pos-root "$POS_ROOT")
[ -n "$NEG_ROOT" ] && extra+=(--neg-root "$NEG_ROOT")

failed=()
for k in "${FOLDS[@]}"; do
    log="$OUT_DIR/fold$k/train.log"
    mkdir -p "$(dirname "$log")"
    echo "=== $(date '+%F %T')  fold $k on $DEVICE  -> $log ==="

    # tee, not just a redirect, so the pane stays watchable while the run is
    # also on disk for later. stderr comes along, which means tqdm's redraws
    # land in the log too -- noisy to read, harmless to keep.
    python scripts/cross_validate.py \
        --data-dir "$DATA_DIR" \
        --out-dir "$OUT_DIR" \
        --fold "$k" \
        --split-seed "$SPLIT_SEED" \
        --wandb-name-prefix "$NAME_PREFIX" \
        -- \
        --device "$DEVICE" \
        --n-patches "$N_PATCHES" \
        ${extra[@]+"${extra[@]}"} 2>&1 | tee "$log"

    rc=${PIPESTATUS[0]}
    # Carry on to the next fold rather than idling the GPU on one failure; the
    # bad folds are named again at the end and the exit status reflects them.
    if [ "$rc" -ne 0 ]; then
        echo "!!! fold $k failed (exit $rc); continuing with the rest"
        failed+=("$k")
    fi
done

echo "=== $(date '+%F %T')  done: folds ${FOLDS[*]} on $DEVICE ==="
if [ ${#failed[@]} -gt 0 ]; then
    echo "failed folds: ${failed[*]}"
    exit 1
fi
