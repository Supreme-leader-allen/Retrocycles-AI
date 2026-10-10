#!/usr/bin/env bash
# Final evaluation of every trained condition: vs random, vs heuristic, vs the
# hand-coded split team, and head-to-head against the main run. All rows go to
# $OUT/metrics/<label>.csv with run_label = <label>. A failed matchup is logged to
# logs/FAILED.txt and the rest still run. Matchups already evaluated are skipped, so
# re-running after an interruption only does what's missing.
set -uo pipefail
source "$(dirname "$0")/params.sh"
evaluate() {  # evaluate <label> <a> <b>
    local label="$1" a="$2" b="$3"
    if [ -f "$OUT/metrics/$label.csv" ]; then echo ">>> $label already evaluated, skipping"; return; fi
    # evaluate.py writes its CSV only after every round has finished, so a crash leaves no file
    python evaluate.py --a "$a" --b "$b" --label "$label" --rounds "$EVAL_ROUNDS" \
        || { rm -f "$OUT/metrics/$label.csv"; note_failure "evaluate $label"; }
}
for ck in "$OUT"/checkpoints/*/latest.pt; do
    run=$(basename "$(dirname "$ck")")
    for opp in random heuristic split; do
        evaluate "final_${run}_vs_${opp}" "$ck" "$opp"
    done
    if [[ "$run" != main_* ]] && [ -f "$OUT/checkpoints/main_s0/latest.pt" ]; then
        evaluate "final_main_s0_vs_${run}" "$OUT/checkpoints/main_s0/latest.pt" "$ck"
    fi
done
