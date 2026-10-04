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

| Rule | Implementation |
|---|---|
| Movement | grid; each decision: straight / turn left / turn right; 2 ticks per decision |
| Crashing | entering any occupied cell (rim, any trail, any head) kills; two cycles entering the same cell both die |
| Speed | 1 cell/decision normally; riding alongside a wall accelerates to 2 cells/decision ("grinding") |
| Explosions | every crash (including head-ons) destroys all trail cells within 2 cells of the crash point, opening holes in walls; the rim is never broken |
| Humanlike limits | the policy sees the game 2 decisions late (`reaction_delay`); optional `turn_cooldown` between turns (off by default) |
| Trails | each cell lasts `trail_ticks` (200); a dead cycle's trail vanishes 30 ticks after death |
| Fortress | each team has a circular zone. If attackers outnumber defenders inside it, it is captured at 1/30 per step per extra attacker (one undefended attacker: 30 steps ~ 3 s); with equal numbers or no attackers it drains (empty from full in ~1.5 s). Time scale: 1 decision ~ 0.1 s |
| Winning | conquer the enemy zone or eliminate every enemy cycle; 500 decisions = draw |

### Simplifications vs real Retrocycles
Grid movement instead of continuous; no rubber, no brakes; two teams only;
wall acceleration only from walls directly beside the head. These are stated
limitations for the paper, not bugs. All are in `tron/config.py`.

## Method

- **Parameter sharing:** one network controls all 14 cycles; observations are egocentric.
- **Observation:** 21x21 egocentric map (walls, own/team/enemy trails and heads, zones)
  + every other cycle's relative position, heading, speed and zone + conquest progress + free-space rays.
- **Reward:** +1 win / -1 loss shared by the whole team, plus annealed shaping
  (conquest progress, kills, individual death penalty) that fades to 0 over 4e8 samples.
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

Reference point: a hand-coded team that splits roles (`split`) beats an all-attack
heuristic team ~98% of the time, every win by conquest. That shows division of labour pays off in this game,
so it is something a learning team has a reason to discover.
