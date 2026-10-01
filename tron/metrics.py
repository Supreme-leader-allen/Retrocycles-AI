"""
Per-round coordination metrics -> CSV (one row per team per finished round).

Accumulated on-device every step (vectorized over rounds) and only copied to
Python when a round ends, so it costs almost nothing during training.

Coordination
- avg_teammate_dist        mean distance between pairs of living teammates (cells). Spread vs clumping.
- role_entropy             normalised entropy of where living teammates are: own zone / midfield /
                           enemy zone. 0 = everyone in the same place, 1 = evenly split roles.
- frac_defending           share of living-cycle steps spent inside your own zone
- frac_attacking           share of living-cycle steps spent inside the enemy zone
- multi_attack_rate        share of steps with 2+ teammates in the enemy zone at once (coordinated push)
- undefended_rate          share of steps where your zone had attackers and no defenders,
                           out of the steps where it had attackers (NaN if never attacked)
- friendly_fire_deaths     crashes into a teammate's trail. Direct coordination failure.
- enemy_walls_blasted      enemy wall cells destroyed by this team's crash explosions (breaching,
                           possibly by sacrifice). Overlapping explosions in the same tick can
                           count a cell twice.
- breaches_made            this team's cycles whose explosion destroyed at least one enemy wall cell
- breaches_used            of those, how many a teammate drove through within breach_window_ticks
- breach_passes            teammate entries into those destroyed cells (one cycle can pass several)
- breach_passes_by_enemy   enemy entries into holes this team blew in the enemy's walls
                           Compare all of these against baseline_random: some passes happen by chance.
- role_specialization      do individual cycles keep DIFFERENT roles? For each cycle that lived
                           >= 10 steps: (steps in enemy zone - steps in own zone) / steps alive.
                           This is the std of that number across teammates. 0 = every cycle
                           behaves the same; high = some cycles attack while others defend.

Outcome / mechanics
- result, end_reason, round_steps, kills, deaths_*, survival, avg_speed, wall_ride_frac,
  max_enemy_progress
- turn_rate (turns per living cycle per step), turns_blocked (turns ignored by turn_cooldown)
"""
import csv
import math
import os

import torch

from .env import CAUSE_RIM, CAUSE_HEADON, REASON_CONQUEST, REASON_ELIMINATION, REASON_TIMEOUT

REASONS = {REASON_CONQUEST: "conquest", REASON_ELIMINATION: "elimination", REASON_TIMEOUT: "timeout"}

COLUMNS = [
    "run_label", "samples", "team", "controller", "opponent", "result", "end_reason", "round_steps",
    "avg_teammate_dist", "role_entropy", "frac_defending", "frac_attacking", "multi_attack_rate",
    "undefended_rate", "friendly_fire_deaths", "enemy_walls_blasted",
    "breaches_made", "breaches_used", "breach_passes", "breach_passes_by_enemy",
    "kills", "deaths_self", "deaths_enemy", "deaths_rim", "deaths_headon",
    "survival", "avg_speed", "wall_ride_frac", "max_enemy_progress",
    "role_specialization", "turn_rate", "turns_blocked",
]

_FIELDS = ["steps", "dist_sum", "dist_cnt", "ent_sum", "ent_cnt", "defend", "attack", "alive_steps",
           "multi_attack", "attacked", "undefended", "ff", "self", "enemy", "rim", "headon", "kills",
           "alive_frac", "speed_sum", "wall", "max_prog", "blast", "turns", "blocked"]
_AGENT_FIELDS = ["own", "foe", "alive"]
MIN_STEPS_FOR_ROLE = 10


def _std(x):
    return float(x.std(unbiased=False)) if x.numel() >= 2 else float("nan")


class MetricsTracker:
    def __init__(self, env, csv_path=None, run_label="run"):
        self.env = env
        self.csv_path = csv_path
        self.run_label = run_label
        self.N, self.T = env.N, env.T
        self.acc = {f: torch.zeros((env.N, 2), device=env.device) for f in _FIELDS}
        self.agent_acc = {f: torch.zeros((env.N, env.A), device=env.device) for f in _AGENT_FIELDS}
        # controller name per (env, team); the trainer overwrites entries for opponent-pool envs
        self.controllers = [["policy", "policy"] for _ in range(env.N)]
        self.samples = 0
        self.rows = []          # rows emitted since the last flush (also used for in-process summaries)
        if csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            if not os.path.exists(csv_path):
                with open(csv_path, "w", newline="") as f:
                    csv.writer(f).writerow(COLUMNS)

    def update(self, info: dict):
        env, N, T = self.env, self.N, self.T
        a = self.acc
        alive_b = info["alive_before"].view(N, 2, T)            # who acted this step
        alive = info["alive"].view(N, 2, T)
        zone = info["zone_of"].view(N, 2, T)
        team_idx = torch.arange(2, device=env.device).view(1, 2, 1)
        n_alive = alive.sum(2).float()
        active = alive_b.any(2).float()                          # team still in the round this step

        a["steps"] += active
        # spacing
        pos = env.pos.view(N, 2, T, 2).float()
        d = torch.cdist(pos, pos)                                # (N, 2, T, T)
        pair = alive.unsqueeze(3) & alive.unsqueeze(2)
        pair &= torch.triu(torch.ones(T, T, dtype=torch.bool, device=env.device), 1)
        npair = pair.sum((2, 3)).float()
        a["dist_sum"] += torch.where(npair > 0, (d * pair).sum((2, 3)) / npair.clamp(min=1), torch.zeros_like(npair))
        a["dist_cnt"] += (npair > 0).float()
        # roles
        in_own = alive & (zone == team_idx)
        in_foe = alive & (zone == 1 - team_idx)
        c_own, c_foe = in_own.sum(2).float(), in_foe.sum(2).float()
        c_mid = n_alive - c_own - c_foe
        p = torch.stack([c_own, c_mid, c_foe], -1) / n_alive.clamp(min=1).unsqueeze(-1)
        ent = -(p * torch.log(p.clamp(min=1e-9))).sum(-1) / math.log(3)
        multi = n_alive >= 2
        a["ent_sum"] += torch.where(multi, ent, torch.zeros_like(ent))
        a["ent_cnt"] += multi.float()
        a["defend"] += c_own
        a["attack"] += c_foe
        a["alive_steps"] += n_alive
        a["multi_attack"] += (c_foe >= 2).float()
        attacked = info["attackers"] > 0
        a["attacked"] += attacked.float()
        a["undefended"] += (attacked & (info["defenders"] == 0)).float()
        # deaths by cause
        died = info["died"].view(N, 2, T)
        cause = info["cause"].view(N, 2, T)
        cause_team = env.team[cause.clamp(min=0)]
        own_id = env.agent_ids.view(1, 2, T)
        trail = died & (cause >= 0)
        a["self"] += (trail & (cause == own_id)).sum(2).float()
        a["ff"] += (trail & (cause != own_id) & (cause_team == team_idx)).sum(2).float()
        enemy_hit = trail & (cause_team != team_idx)
        a["enemy"] += enemy_hit.sum(2).float()
        a["rim"] += (died & (cause == CAUSE_RIM)).sum(2).float()
        a["headon"] += (died & (cause == CAUSE_HEADON)).sum(2).float()
        a["blast"] += info["blasted"].view(N, 2, T).sum(2).float()
        a["turns"] += info["turned"].view(N, 2, T).sum(2).float()
        a["blocked"] += info["turns_blocked"].view(N, 2, T).sum(2).float()
        ag = self.agent_acc
        ag["own"] += in_own.view(N, -1).float()
        ag["foe"] += in_foe.view(N, -1).float()
        ag["alive"] += alive.view(N, -1).float()
        # kills: enemy deaths on our trails
        killer_team = torch.where(enemy_hit, cause_team, torch.full_like(cause_team, -1))
        for k in range(2):
            a["kills"][:, k] += (killer_team == k).sum((1, 2)).float()
        # mechanics
        a["alive_frac"] += n_alive / T * active
        spd = env.speed.view(N, 2, T).float() / 100.0 * env.cfg.ticks_per_step  # cells per step
        a["speed_sum"] += (spd * alive).sum(2)
        a["wall"] += (env.near_wall.view(N, 2, T) & alive).sum(2).float()
        a["max_prog"] = torch.maximum(a["max_prog"], info["progress"].flip(1))

        done = info["done"]
        if done.any():
            self._emit(done, info)

    def _emit(self, done, info):
        idx = done.nonzero(as_tuple=True)[0]
        snap = {k: v[idx].cpu() for k, v in self.acc.items()}
        ag = {k: v[idx].view(-1, 2, self.T).cpu() for k, v in self.agent_acc.items()}
        lived = ag["alive"] >= MIN_STEPS_FOR_ROLE
        pref = (ag["foe"] - ag["own"]) / ag["alive"].clamp(min=1)
        br = {k: info[k][idx].view(-1, 2, self.T).sum(2).float().cpu()
              for k in ("breach_made", "breach_used", "breach_passes", "breach_enemy_passes")}
        winner = info["winner"][idx].cpu()
        reason = info["reason"][idx].cpu()
        steps = info["steps"][idx].cpu()
        for row_i, n in enumerate(idx.tolist()):
            for k in range(2):
                g = lambda f: float(snap[f][row_i, k])
                w = int(winner[row_i])
                res = "draw" if w < 0 else ("win" if w == k else "loss")
                alive_steps = max(g("alive_steps"), 1.0)
                self.rows.append({
                    "run_label": self.run_label, "samples": self.samples, "team": k,
                    "controller": self.controllers[n][k], "opponent": self.controllers[n][1 - k],
                    "result": res, "end_reason": REASONS.get(int(reason[row_i]), "none"),
                    "round_steps": int(steps[row_i]),
                    "avg_teammate_dist": g("dist_sum") / g("dist_cnt") if g("dist_cnt") else float("nan"),
                    "role_entropy": g("ent_sum") / g("ent_cnt") if g("ent_cnt") else float("nan"),
                    "frac_defending": g("defend") / alive_steps,
                    "frac_attacking": g("attack") / alive_steps,
                    "multi_attack_rate": g("multi_attack") / max(g("steps"), 1.0),
                    "undefended_rate": g("undefended") / g("attacked") if g("attacked") else float("nan"),
                    "friendly_fire_deaths": g("ff"),
                    "enemy_walls_blasted": g("blast"),
                    "breaches_made": float(br["breach_made"][row_i, k]),
                    "breaches_used": float(br["breach_used"][row_i, k]),
                    "breach_passes": float(br["breach_passes"][row_i, k]),
                    "breach_passes_by_enemy": float(br["breach_enemy_passes"][row_i, k]),
                    "kills": g("kills"), "deaths_self": g("self"), "deaths_enemy": g("enemy"),
                    "deaths_rim": g("rim"), "deaths_headon": g("headon"),
                    "survival": g("alive_frac") / max(g("steps"), 1.0),
                    "avg_speed": g("speed_sum") / alive_steps,
                    "wall_ride_frac": g("wall") / alive_steps,
                    "max_enemy_progress": g("max_prog"),
                    "role_specialization": _std(pref[row_i, k][lived[row_i, k]]),
                    "turn_rate": g("turns") / alive_steps,
                    "turns_blocked": g("blocked"),
                })
        for v in self.acc.values():
            v[idx] = 0.0
        for v in self.agent_acc.values():
            v[idx] = 0.0

    def flush(self):
        """Append pending rows to the CSV and return them."""
        rows, self.rows = self.rows, []
        if self.csv_path and rows:
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                for r in rows:
                    w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
        return rows
