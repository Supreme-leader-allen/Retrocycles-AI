#!/usr/bin/env bash
# Everything, in order. Same script locally (smoke test) and on the pod (real run):
#   local :  NUM_ENVS=64   SAMPLES=2e6 bash experiments/run_all.sh
#   pod   :  NUM_ENVS=2048 SAMPLES=2e9 bash experiments/run_all.sh
# CORE_ONLY=1 skips the ablations (02-06). SKIP_PBT=1 skips PBT (07).
set -euo pipefail
DIR="$(dirname "$0")"
source "$DIR/params.sh"
bash "$DIR/00_baseline.sh"
bash "$DIR/01_main.sh"
if [ "${CORE_ONLY:-0}" != "1" ]; then
    bash "$DIR/02_no_credit.sh"
    bash "$DIR/03_no_anneal.sh"
    bash "$DIR/04_partial_info.sh"
    bash "$DIR/05_no_pool.sh"
    bash "$DIR/06_no_humanlike.sh"
fi
if [ "${SKIP_PBT:-0}" != "1" ]; then
    bash "$DIR/07_pbt.sh"     # PBT_MEMBERS x the compute of one run; SKIP_PBT=1 to leave it out
fi
bash "$DIR/99_evaluate.sh"
echo
echo "================================================================"
echo " ALL DONE. Pack the results (everything except the big checkpoints) and pull them back:"
echo "   tar czf results.tgz -C $OUT metrics logs heatmaps replays \$(cd $OUT && ls -d checkpoints/*/run_info.json checkpoints/*/latest.pt)"
echo "   runpodctl send results.tgz"
echo " AND STOP THE POD."
echo "================================================================"
