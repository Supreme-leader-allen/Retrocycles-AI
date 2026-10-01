#!/usr/bin/env bash
# Ablation: a cycle's episode ends when IT dies, so it never receives the team's
# win/loss after crashing ("selfish credit"). Isolates shared credit assignment.
set -euo pipefail
source "$(dirname "$0")/params.sh"
for s in $(seed_list); do
    train "no_credit_s$s" --train num_envs="$NUM_ENVS" seed="$s" credit_after_death=false
done
