#!/usr/bin/env bash
# Everything, in order. Same script locally (smoke test) and on the pod (real run):
#   pod :  NUM_ENVS=1024 SAMPLES=2e9 bash experiments/run_all.sh
# CORE_ONLY=1 skips the ablations (02-06, 08). SKIP_PBT=1 skips PBT (07).
# If one condition crashes, it is logged to $OUT/logs/FAILED.txt and the others still run.
# Re-running the same command resumes: finished runs are skipped, unfinished ones continue.
set -uo pipefail
DIR="$(dirname "$0")"
source "$DIR/params.sh"
step() {
    local script="$1"
    echo; echo "================ $script  ($(date '+%F %T'))"
    bash "$DIR/$script" || note_failure "$script exited with an error"
}
step 00_baseline.sh
step 01_main.sh
if [ "${CORE_ONLY:-0}" != "1" ]; then
    step 02_no_credit.sh
    step 03_no_anneal.sh
    step 04_partial_info.sh
    step 05_no_pool.sh
    step 06_no_humanlike.sh
    step 08_no_agent_id.sh
fi
if [ "${SKIP_PBT:-0}" != "1" ]; then
    step 07_pbt.sh     # PBT_MEMBERS x the compute of one run; SKIP_PBT=1 to leave it out
fi
step 99_evaluate.sh
echo
echo "================================================================"
if [ -s "$OUT/logs/FAILED.txt" ]; then
    echo " FINISHED WITH ERRORS -- see $OUT/logs/FAILED.txt:"
    cat "$OUT/logs/FAILED.txt"
else
    echo " ALL DONE, no errors."
fi
echo " Pack the results (everything except the big checkpoints) and pull them back:"
echo "   cd $OUT && tar czf /workspace/results.tgz metrics logs heatmaps replays checkpoints/*/run_info.json"
echo " AND STOP THE POD."
echo "================================================================"
