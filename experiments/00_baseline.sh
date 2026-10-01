#!/usr/bin/env bash
# Control conditions: no learning at all.
#   baseline_random      random vs random          -> what the coordination metrics look like by chance
#   heuristic_mirror     heuristic vs heuristic    -> a competent but uncoordinated team
#   split_vs_heuristic   hand-coded role split     -> what a simple deliberate division of labour looks like
set -euo pipefail
source "$(dirname "$0")/params.sh"
python evaluate.py --a random    --b random    --team-size "$TEAM_SIZE" --label baseline_random    --rounds "$EVAL_ROUNDS"
python evaluate.py --a heuristic --b heuristic --team-size "$TEAM_SIZE" --label heuristic_mirror   --rounds "$EVAL_ROUNDS"
python evaluate.py --a split     --b heuristic --team-size "$TEAM_SIZE" --label split_vs_heuristic --rounds "$EVAL_ROUNDS"
