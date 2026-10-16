#!/usr/bin/env bash
# Replication: extra training seeds of the key conditions, then the final evaluation.
#   SAMPLES=2e9 bash experiments/10_replicate.sh
# REPLICATE lists the condition scripts (default: main, no_credit, no_anneal); SEEDS=3 means
# seeds 0,1,2 -- seed 0 already exists and is skipped instantly, so seeds 1 and 2 train.
# Re-running resumes like run_all.sh; failures go to logs/FAILED.txt and the rest continue.
set -uo pipefail
DIR="$(dirname "$0")"
export SEEDS="${SEEDS:-3}"
source "$DIR/params.sh"
REPLICATE="${REPLICATE:-01_main.sh 02_no_credit.sh 03_no_anneal.sh}"
for script in $REPLICATE; do
    echo; echo "================ $script, seeds 0..$((SEEDS - 1))  ($(date '+%F %T'))"
    bash "$DIR/$script" || note_failure "$script (replication) exited with an error"
done
echo; echo "================ 99_evaluate.sh  ($(date '+%F %T'))"
bash "$DIR/99_evaluate.sh" || note_failure "99_evaluate.sh exited with an error"
echo
echo "================================================================"
if [ -s "$OUT/logs/FAILED.txt" ]; then echo " FINISHED WITH ERRORS -- see $OUT/logs/FAILED.txt"; cat "$OUT/logs/FAILED.txt"
else echo " REPLICATION DONE, no errors."; fi
echo " Pack the new results:"
echo "   cd $OUT && tar czf /workspace/results_replication.tgz metrics logs heatmaps replays checkpoints/*/run_info.json \$(for d in checkpoints/*/; do ls -t \$d/ckpt_*.pt 2>/dev/null | head -1; done)"
echo "================================================================"
