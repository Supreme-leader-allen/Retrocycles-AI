#!/usr/bin/env bash
# Ablation: partial observability. Enemy cycles further than 15 cells away are
# hidden from the vector observation (teammates always visible, like a team
# radar). Isolates the effect of information on coordination.
set -euo pipefail
source "$(dirname "$0")/params.sh"
for s in $(seed_list); do
    train "partial_info_s$s" --train num_envs="$NUM_ENVS" seed="$s" --game vis_radius=15
done
