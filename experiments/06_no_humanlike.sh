#!/usr/bin/env bash
# Ablation: humanlike limits OFF. The policy sees the game instantly (reaction_delay=0)
# instead of 2 decisions late. Isolates the effect of reaction time on coordination.
set -euo pipefail
source "$(dirname "$0")/params.sh"
for s in $(seed_list); do
    train "no_humanlike_s$s" --train num_envs="$NUM_ENVS" seed="$s" --game reaction_delay=0 turn_cooldown=0
done
