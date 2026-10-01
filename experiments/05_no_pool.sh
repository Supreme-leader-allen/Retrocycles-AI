#!/usr/bin/env bash
# Ablation: pure self-play, no opponent pool of past snapshots.
set -euo pipefail
source "$(dirname "$0")/params.sh"
for s in $(seed_list); do
    train "no_pool_s$s" --train num_envs="$NUM_ENVS" seed="$s" pool_frac=0
done
