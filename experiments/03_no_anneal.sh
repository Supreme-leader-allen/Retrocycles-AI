#!/usr/bin/env bash
# Ablation: dense shaping (conquest / kill / death) stays at full strength forever
# instead of annealing to 0. Isolates reward annealing.
set -euo pipefail
source "$(dirname "$0")/params.sh"
for s in $(seed_list); do
    train "no_anneal_s$s" --train num_envs="$NUM_ENVS" seed="$s" --reward anneal_samples=0
done
