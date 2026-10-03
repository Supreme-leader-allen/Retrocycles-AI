#!/usr/bin/env bash
# All experiment parameters, in three layers. Sourced by every script here.
# You only ever edit LAYER 3.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# =============================================================================
# LAYER 1 — FROZEN. Changing these changes the research question or breaks
# comparisons between runs. Listed so you know not to touch them.
# =============================================================================
#   team size          7v7 Fortress (FortressConfig defaults)   -> different game
#   arena / rules      FortressConfig defaults                   -> runs not comparable
#   network            Policy(hidden=256), obs_radius=10         -> checkpoints incompatible
#   win reward         +1 / -1, team-shared                      -> this IS the credit-assignment signal
#   humanlike limits   reaction_delay=2 decisions, turn_cooldown=0 (only 06 changes them)
TEAM_SIZE=7

# =============================================================================
# LAYER 2 — MEASURED, not chosen. Run `python benchmark.py` on the pod first.
# =============================================================================
# Parallel rounds simulated on the GPU. Largest value before samples/sec stops
# improving (or memory runs out).
NUM_ENVS="${NUM_ENVS:?run 'python benchmark.py' first, then: export NUM_ENVS=<best num_envs>}"

# =============================================================================
# LAYER 3 — YOURS. The only layer you edit.
# =============================================================================
# Agent-steps per training run (1 agent-step = one cycle making one decision).
#   local smoke test :  SAMPLES=2e6      (does the pipeline work end to end?)
#   real run         :  SAMPLES=2e9      (time it with benchmark.py's "train sps")
SAMPLES="${SAMPLES:-2e6}"

# Rounds played per final evaluation matchup.
EVAL_ROUNDS="${EVAL_ROUNDS:-1024}"

# Seeds per training condition. More seeds = error bars across training runs,
# not just across rounds. 1 is fine for a first pass; 3 is better for the paper.
SEEDS="${SEEDS:-1}"

# Where results go. /workspace is RunPod's persistent volume. Not /workspace/output: that is
# where the Rocket League project writes, and the two may share a volume.
if [ -d /workspace ]; then
    OUT="${OUT:-/workspace/retrocycles_output}"
else
    OUT="${OUT:-$REPO_ROOT/output}"
fi
export OUT
mkdir -p "$OUT/checkpoints" "$OUT/metrics" "$OUT/logs"

# train <label> [extra train.py args...]
# Resumes automatically if the run already has a checkpoint (pod restarted, etc).
train() {
    local label="$1"; shift
    if [ -f "$OUT/checkpoints/$label/latest.pt" ]; then
        echo ">>> $label: checkpoint exists, resuming"
        python train.py --label "$label" --out "$OUT" --samples "$SAMPLES" --resume \
            --train num_envs="$NUM_ENVS" 2>&1 | tee -a "$OUT/logs/$label.log"
    else
        python train.py --label "$label" --out "$OUT" --team-size "$TEAM_SIZE" --samples "$SAMPLES" \
            "$@" 2>&1 | tee -a "$OUT/logs/$label.log"
    fi
}

seed_list() { seq 0 $(( SEEDS - 1 )); }
