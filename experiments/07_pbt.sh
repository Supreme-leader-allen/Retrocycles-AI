#!/usr/bin/env bash
# Population-based training: PBT_MEMBERS members, each trained for $SAMPLES agent-steps
# (so PBT_MEMBERS x the compute of one main run), hyperparameters evolved every
# generation from cross-play fitness. The best member ends up in
# checkpoints/pbt_s<seed>/latest.pt and is evaluated by 99_evaluate.sh like every other run.
# Re-running resumes.
set -euo pipefail
source "$(dirname "$0")/params.sh"
PBT_MEMBERS="${PBT_MEMBERS:-4}"
PBT_GENERATIONS="${PBT_GENERATIONS:-10}"
for s in $(seed_list); do
    python pbt.py --label "pbt_s$s" --out "$OUT" --team-size "$TEAM_SIZE" --samples "$SAMPLES" \
        --members "$PBT_MEMBERS" --generations "$PBT_GENERATIONS" --seed "$s" \
        --train num_envs="$NUM_ENVS" 2>&1 | tee -a "$OUT/logs/pbt_s$s.log"
done
