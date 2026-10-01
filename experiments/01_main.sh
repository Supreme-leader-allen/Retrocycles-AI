#!/usr/bin/env bash
# Main result: full method stack.
#   team-shared reward + credit after death + annealed shaping + opponent pool, full information
set -euo pipefail
source "$(dirname "$0")/params.sh"
for s in $(seed_list); do
    train "main_s$s" --train num_envs="$NUM_ENVS" seed="$s"
done
