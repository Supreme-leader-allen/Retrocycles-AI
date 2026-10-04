#!/usr/bin/env bash
# Ablation: cycles do NOT see their own slot number, so the shared policy can only tell
# teammates apart by where they are. Isolates whether identity enables role specialization.
set -euo pipefail
source "$(dirname "$0")/params.sh"
for s in $(seed_list); do
    train "no_agent_id_s$s" --train num_envs="$NUM_ENVS" seed="$s" --game agent_id_obs=false
done
