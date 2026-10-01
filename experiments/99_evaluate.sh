#!/usr/bin/env bash
# Final evaluation of every trained condition: vs random, vs heuristic, vs the
# hand-coded split team, and head-to-head against the main run. All rows go to
# $OUT/metrics/<label>.csv with run_label = <label>.
set -euo pipefail
source "$(dirname "$0")/params.sh"
for ck in "$OUT"/checkpoints/*/latest.pt; do
    run=$(basename "$(dirname "$ck")")
    for opp in random heuristic split; do
        python evaluate.py --a "$ck" --b "$opp" --label "final_${run}_vs_${opp}" --rounds "$EVAL_ROUNDS"
    done
    if [[ "$run" != main_* ]] && [ -f "$OUT/checkpoints/main_s0/latest.pt" ]; then
        python evaluate.py --a "$OUT/checkpoints/main_s0/latest.pt" --b "$ck" \
            --label "final_main_s0_vs_${run}" --rounds "$EVAL_ROUNDS"
    fi
done
