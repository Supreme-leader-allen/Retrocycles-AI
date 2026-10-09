# Retrocycles AI: emergent coordination in 7v7 Fortress

Multi-agent reinforcement learning on a custom, GPU-vectorized simulator of
Retrocycles / Armagetron **Fortress** (team light-cycles with base zones).

**Research question:** can teams of agents trained only on a shared win/loss
signal develop coordinated behaviour (role splitting into attackers and
defenders, coordinated pushes, avoiding teammates' walls), and which training
choices are responsible for it?

## Layout

```
tron/config.py     game rules + sizes (FortressConfig)
tron/env.py        vectorized simulator: N rounds x 14 cycles stepped in one call
tron/rewards.py    team-shared reward + annealed shaping
tron/metrics.py    per-round coordination metrics -> CSV
tron/bots.py       random / heuristic / split-role scripted baselines
tron/model.py      shared policy/value network (CNN on local map + MLP on all cycles)
tron/ppo.py        PPO self-play trainer with opponent pool
tron/play.py       match play between any two controllers
train.py           train a run          evaluate.py   matchups -> metrics CSV
render.py          replay one round as a GIF
benchmark.py       pick num_envs for a machine
experiments/       the experiment matrix (same scripts locally and on RunPod)
pbt.py             population-based training
tron/runner.py     training loop + all logging (see DATA.md)
DATA.md            every file and column that gets recorded
tests/             hand-built game states with known outcomes, checked by execution
```

## Quick start (local, CPU)

```bash
pip install -r requirements.txt
python -m pytest -q                       # 66 tests: rules, observations, trainer, PBT
python render.py --a split --b heuristic --gif replays/bots.gif
python train.py --label try --team-size 2 --samples 3e5 --train num_envs=64 rollout_len=32 minibatch=8192
```

CPU training is only for checking that things run; real training needs a GPU.

## Real runs (RunPod)

```bash
git clone <repo> && cd "Retrocycles AI"
bash experiments/setup_pod.sh             # installs, checks CUDA, runs tests, benchmarks
export NUM_ENVS=<best from benchmark>
SAMPLES=2e9 bash experiments/run_all.sh   # baseline, main, 4 ablations, final evaluation
```

`train.py` resumes automatically from `latest.pt` if a run is interrupted
(the experiment scripts do this for you). Results: `$OUT/metrics/*.csv`.

## The game (what the simulator models)

Every value comes from Armagetron Advanced, which Retrocycles is based on: the fortress server
config shipped with the game (`fortress_soccer.cfg`, `cvs_test/fortress_physics.cfg`), its map
(`Z-Man/fortress/for_old_clients-0.1.0.aamap.xml`), the engine defaults in its source code, and the
wiki's Fortress page. In-game measurements (end to end 16.55 s, zone to zone 10.35 s) match the
simulator's 16.7 s and 10.7 s. All settings, with sources, are in `tron/config.py`.

Scale: **1 cell = 3 m, 1 decision = 0.1 s** (30 m/s = one cell per decision), 3 physics ticks per decision.

| Rule | Implementation |
|---|---|
| Map | 500 x 500 m (167 x 167 cells); zones radius 40 m, centres 50 m from the rim |
| Spawn | V ("wingmen") formation beside your zone centre, facing the enemy (SPAWN_WINGMEN_BACK 2.2 m, SIDE 2.75 m) |
| Movement | straight / turn left / turn right each decision (4 arena axes, CYCLE_DELAY 0.1 s) |
| Speed | 30 m/s; walls within 6 m to the side accelerate you (CYCLE_ACCEL 20, Armagetron's 1/distance falloff; the rim does not); boosts decay by 10% of the excess per second; max 90 m/s |
| Rubber | 5 m: a cycle driving into a wall is held in front of it and burns rubber equal to the distance it would have driven; dies when it runs out; refills in 10 s. Head-on collisions kill at once |
| Walls | 400 m long (by distance driven); a dead cycle's walls stay up 8 s |
| Explosions | every death destroys trail walls within 4 m of the crash point (never the rim) |
| Fortress | capture progress per second = 0.3 x attackers - 0.2 x defenders - 0.1 (1 attacker alone: 5 s; 1 vs 1: never; 2 vs 0: 2 s; 2 vs 1: 3.3 s; 2 vs 2: 10 s) |
| Winning | capture the enemy zone or eliminate every enemy cycle; 120 s = draw |
| Humanlike limits | the policy sees the game 2 decisions (0.2 s) late; optional extra turn cooldown (off) |

### Simplifications vs real Retrocycles
3 m grid instead of continuous movement; rubber holds you in front of a wall rather than slowing you
as you approach; no brakes; two teams only; max 90 m/s. These are stated limitations for the paper.

## Method

- **Parameter sharing:** one network controls all 14 cycles; observations are egocentric.
- **Observation:** 21x21 egocentric map (walls, own/team/enemy trails and heads, zones)
  + every other cycle's relative position, heading, speed and zone + conquest progress + free-space rays.
- **Reward:** +1 win / -1 loss shared by the whole team, a permanent individual death penalty (-0.5),
  plus conquest and kill shaping that fades to 0 over 4e8 samples.
- **Credit after death:** a cycle's episode is the whole round, so a cycle that dies still
  receives its team's eventual result (actions that helped the team before dying get credit).
- **Self-play + opponent pool + scripted opponents:** 25% of rounds are played against frozen past snapshots and
  20% against the heuristic (all-attack rush) and split-role bots, so the policy also meets strategies its own
  self-play never produces (without them it learned to leave its base empty).
- **Humanlike limits:** 2-decision reaction delay on the policy's observations.
- **Agent identity:** each cycle sees its slot number (1-7), so the shared policy can assign fixed roles.
- **PBT (optional condition):** population of 4; every generation the bottom quarter copies a top member
  and perturbs lr / entropy / conquest / kill weights by ×0.8-1.2; fitness = cross-play win rate.

## Experiment matrix

| Script | Label | Varies | Compare against |
|---|---|---|---|
| `00_baseline.sh` | `baseline_random`, `heuristic_mirror`, `split_vs_heuristic` | no learning (controls) | — |
| `01_main.sh` | `main_s<seed>` | full method | baseline |
| `02_no_credit.sh` | `no_credit_s<seed>` | cycle's episode ends at its own death | main |
| `03_no_anneal.sh` | `no_anneal_s<seed>` | shaping never fades | main |
| `04_partial_info.sh` | `partial_info_s<seed>` | enemies beyond 15 cells hidden | main |
| `05_no_pool.sh` | `no_pool_s<seed>` | pure self-play | main |
| `06_no_humanlike.sh` | `no_humanlike_s<seed>` | instant reactions (`reaction_delay=0`) | main |
| `08_no_agent_id.sh` | `no_agent_id_s<seed>` | cycles don't see their slot number | main |
| `07_pbt.sh` | `pbt_s<seed>` (members `pbt_s<seed>_m<k>`) | population-based training: 4 members, hyperparameters evolved from cross-play fitness (4× compute) | main |
| `99_evaluate.sh` | `final_<run>_vs_<opp>` | every run vs random / heuristic / split, + head-to-head vs main | |

## Metrics

Full reference in **DATA.md**. Per round (one CSV row per team per round):

Coordination: `avg_teammate_dist`, `role_entropy`, `frac_defending`, `frac_attacking`,
`multi_attack_rate`, `undefended_rate`, `friendly_fire_deaths`.
Wall breaching: `enemy_walls_blasted`, `breaches_made` (deaths that blew a hole in an enemy wall),
`breaches_used` (holes a teammate then drove through within 60 ticks), `breach_passes`, `breach_passes_by_enemy`.
Outcome/mechanics: `result`, `end_reason`, `round_steps`, `kills`, `deaths_*`, `survival`,
`avg_speed`, `wall_ride_frac`, `max_enemy_progress`. Definitions are in `tron/metrics.py`.

Reference points (7v7, 128 rounds): the all-attack `heuristic` team (fanning out into flank lanes)
beats the hand-coded wiki-positions `split` team ~94% of the time, all by conquest. Naive role-splitting
loses to a coordinated rush, so a learned team has to discover *effective* defence, not just presence.
