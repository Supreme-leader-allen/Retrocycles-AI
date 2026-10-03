# What gets recorded

Everything lives under `$OUT` (default `output/`, `/workspace/retrocycles_output` on RunPod).
`<label>` is the run name (`main_s0`, `no_credit_s0`, `pbt_s0_m2`, ...). It is also the
`run_label` column in the CSVs, so you can concatenate all CSVs and group by it.

| Path | Written | Use it for |
|---|---|---|
| `logs/<label>_train.csv` | every training iteration | learning curves, debugging training |
| `metrics/<label>.csv` | every training round that ends | coordination over the course of training |
| `metrics/<label>_eval.csv` | every `eval_every` iterations (default 25) | learning curves vs fixed opponents: the cleanest "is it getting better" signal |
| `metrics/<other>.csv` from `evaluate.py` | final evaluation, baselines | **the main results tables / t-tests** |
| `metrics/<pbt>_crossplay.csv` | every PBT generation | PBT tournament results |
| `logs/<pbt>_generations.jsonl` | every PBT generation | hyperparameter evolution, fitness, lineage plots |
| `heatmaps/<label>/vs_heuristic_<samples>.npz` | every eval | where cycles drive and die (figure) |
| `replays/<label>/<samples>_vs_heuristic.gif` | every `replay_every` iterations (default 100) + iteration 1 | watching behaviour change; poster/presentation clips |
| `checkpoints/<label>/latest.pt`, `ckpt_<samples>.pt` | every `save_every` iterations | resuming; evaluating old checkpoints |
| `checkpoints/<label>/run_info.json` | at start | exact configs, git commit, torch/GPU versions, command line (reproducibility) |
| `checkpoints/<label>/debug_<reason>_<it>.pt` | only on a crash | reproducing a NaN / simulator bug: model, optimizer AND full game state |
| `logs/<label>.log` | stdout of the experiment scripts | |

## Per-round metrics (`metrics/*.csv`): one row per team per round

**Identity:** `run_label`, `samples` (training progress when the round was played), `team` (0 bottom / 1 top),
`controller` / `opponent` (`policy`, `pool`, `heuristic`, `split`, `random`, a checkpoint's run name, `m<k>` for PBT).
**Filter on `controller`** when computing a run's stats: each round has a row for both sides.

**Outcome:** `result` (win/loss/draw), `end_reason` (conquest/elimination/timeout), `round_steps`, `max_enemy_progress`.

**Coordination (the research variables):**

| Column | Meaning | Coordinated team looks like |
|---|---|---|
| `role_entropy` | spread of living teammates across own zone / midfield / enemy zone (0-1) | higher |
| `role_specialization` | std across teammates of (enemy-zone time − own-zone time); do *individual* cycles keep different roles? | higher |
| `frac_defending`, `frac_attacking` | share of living time in own / enemy zone | both > 0 |
| `multi_attack_rate` | share of steps with 2+ teammates in the enemy zone | coordinated pushes |
| `undefended_rate` | share of under-attack steps with no defender home | lower |
| `avg_teammate_dist` | spacing between living teammates | not clumped |
| `friendly_fire_deaths` | crashes into a teammate's trail | lower |
| `enemy_walls_blasted`, `breaches_made` | enemy wall destroyed by this team's crash explosions | |
| `breaches_used`, `breach_passes` | a teammate drove through that hole within 60 ticks | higher `breaches_used / breaches_made` than baselines |
| `breach_passes_by_enemy` | enemies drove through holes in their own wall | |

**Mechanics / debugging:** `kills`, `deaths_self` / `deaths_enemy` / `deaths_rim` / `deaths_headon`, `survival`,
`avg_speed` (cells/decision), `wall_ride_frac`, `turn_rate`, `turns_blocked`.

Reference values (scripted bots, 7v7, 128 rounds): split-role bot `role_specialization` ≈ 0.32,
`frac_defending` ≈ 0.18, `undefended_rate` ≈ 0.46, beats the all-attack heuristic 98% (all by conquest);
the heuristic has `frac_defending` 0 and `undefended_rate` 1.0. `00_baseline.sh` re-measures these.

## Training log (`logs/<label>_train.csv`): one row per iteration

| Group | Columns | What to look for |
|---|---|---|
| progress | `iteration`, `samples`, `sps`, `collect_s`, `update_s`, `gpu_mem_gb` | throughput dropping = something wrong with the pod |
| PPO health | `pg_loss`, `v_loss`, `entropy`, `approx_kl`, `clipfrac`, `grad_norm`, `explained_var`, `param_norm` | entropy collapsing to ~0 early = premature convergence; kl > 0.05 or clipfrac > 0.3 = lr too high; explained_var should rise above 0 |
| values | `value_mean`, `return_mean`, `adv_std_raw`, `mean_reward` | |
| reward parts | `rew_win`, `rew_conquest`, `rew_kill`, `rew_death` (mean per sample, already scaled) | shaping should shrink to 0 as `shaping_scale` anneals; if shaping dwarfs `rew_win`, the agent optimises the shaping |
| behaviour | `act_straight`, `act_left`, `act_right`, `alive_frac`, `turn_rate`, `avg_speed`, `survival` | a policy stuck at ~100% one action = collapse |
| rounds | `rounds`, `round_steps`, `selfplay_draw`, `end_conquest`, `end_elimination`, `end_timeout` | the game strategy shifting from elimination to conquest is a finding |
| **bug detector** | `side0_win` | self-play is the same policy on both sides, so this must hover around 0.5. Persistently far from it = an asymmetry bug |
| deaths | `deaths_self`, `friendly_fire_deaths`, `deaths_enemy`, `deaths_rim`, `deaths_headon`, `kills` | |
| coordination (self-play) | `role_entropy`, `role_specialization`, `frac_defending`, `frac_attacking`, `multi_attack_rate`, `undefended_rate`, `avg_teammate_dist`, `breaches_made`, `breaches_used` | coordination emerging over training |
| opponents | `win_vs_pool`, `pool_size` | win_vs_pool well above 0.5 = still improving over its past self |
| eval (NaN except on eval iterations) | `eval_heuristic_win/draw/role_spec/breach_use`, same for `eval_split_*` | **the main learning curve** |
| settings | `lr`, `ent_coef`, `shaping_scale` | (change over time under PBT / annealing) |

## Heatmaps (`heatmaps/**/*.npz`)

`np.load(f)` gives `visits` and `deaths` (S×S counts) for the policy's cycles in a **canonical frame**:
own zone at the bottom, enemy zone at the top (team 1 is rotated 180°). Also `zone_center`, `zone_radius`,
`samples`. Plot `visits` with `imshow` to show where the team goes; compare early vs late checkpoints.

## Safety nets that stop a run loudly instead of producing quietly wrong data

- **Simulator invariants** (`check_every`, default every 10 iterations), checked on the live training batch:
  rim intact, ids valid, every living cycle sits on its own cell, no two living cycles on one cell,
  speeds and conquest progress in range. Any violation saves `debug_invariant_*.pt` and raises.
- **Non-finite loss** saves `debug_nan_*.pt` and raises.
- **`side0_win`** in the log (see above).
- `tests/` (63 tests) must pass before a run: `python -m pytest -q`.

## Known measurement caveats (state these in the paper)

- Scripted bots read the game state directly, so they have **no reaction delay**; the trained policy plays
  with `reaction_delay=2` decisions. Wins against bots are therefore against a slightly faster-reacting opponent.
- `breaches_used` counts entering a cell that used to be enemy wall; it does not verify the cycle came out
  the far side. Some passes happen by chance: always compare to the baselines.
- Overlapping explosions in the same tick can count a destroyed cell twice in `enemy_walls_blasted`.
- Training-round metrics (`metrics/<label>.csv`) come from self-play against itself and past copies, so
  they drift with the opponent. Use `_eval.csv` and the final `evaluate.py` runs for clean comparisons.
