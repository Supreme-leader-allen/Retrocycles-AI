#!/usr/bin/env bash
# Control conditions: no learning at all. Skipped if already evaluated.
#   baseline_random      random vs random          -> what the coordination metrics look like by chance
#   heuristic_mirror     heuristic vs heuristic    -> a competent all-attack team against itself
#   split_vs_heuristic   wiki positions vs rush    -> a hand-coded role split against the rush
set -uo pipefail
source "$(dirname "$0")/params.sh"
base() {  # base <label> <a> <b>
    [ -f "$OUT/metrics/$1.csv" ] && { echo ">>> $1 already done, skipping"; return; }
    python evaluate.py --a "$2" --b "$3" --team-size "$TEAM_SIZE" --label "$1" --rounds "$EVAL_ROUNDS" \
        || note_failure "baseline $1"
}
base baseline_random    random    random
base heuristic_mirror   heuristic heuristic
base split_vs_heuristic split     heuristic
