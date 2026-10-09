"""
Vectorized Retrocycles Fortress simulator.

All state lives in torch tensors with a leading ``N`` (number of parallel
rounds) dimension, so one ``step()`` call advances every round at once on
CPU or GPU. There is no Python loop over rounds or agents.

Usage
-----
    env = FortressEnv(FortressConfig(team_size=7), num_envs=1024, device="cuda")
    codes, vec = env.observe()
    info = env.step(actions)          # actions: (N, A) long in {0 straight, 1 left, 2 right}
    ... compute rewards / metrics from info ...
    env.reset_done(info["done"])      # start new rounds where one just ended
    codes, vec = env.observe()

``step`` deliberately does NOT auto-reset, so rewards and metrics can read the
terminal state of a round before it is wiped.

Grid encoding (``owner``, int16, shape (N, G, G) with G = arena + 2*pad):
    EMPTY (-1)   free cell
    RIM   (-2)   arena boundary (the whole padding band)
    0..A-1       trail cell laid by that agent (a cycle's current cell is
                 also its trail cell)

Agents 0..T-1 are team 0, T..2T-1 are team 1.
Directions: 0 up (-y), 1 right (+x), 2 down (+y), 3 left (-x).

Death causes (``death_cause``): -1 alive, -2 rim, -3 head-on collision with
another cycle entering the same cell, 0..A-1 the owner of the trail hit.

Physics (real-world values in config.py): rubber holds a cycle in front of a
wall until it is used up; walls within 2 cells to the side accelerate; trail
cells are stamped with the owner's odometer (cells driven) so wall length is
a distance, as in Armagetron.
"""
import torch

from .config import FortressConfig, SPEED_UNIT

EMPTY = -1
RIM = -2
CAUSE_ALIVE = -1
CAUSE_RIM = -2
CAUSE_HEADON = -3

# end-of-round reasons
REASON_NONE, REASON_CONQUEST, REASON_ELIMINATION, REASON_TIMEOUT = 0, 1, 2, 3

# observation cell codes: code = occupancy * 3 + zone
OCC_EMPTY, OCC_RIM, OCC_OWN, OCC_TEAM, OCC_ENEMY, OCC_TEAM_HEAD, OCC_ENEMY_HEAD = range(7)
ZONE_NONE, ZONE_OWN, ZONE_ENEMY = range(3)
NUM_CELL_CODES = 7 * 3

SELF_FEATURES = 17
OTHER_FEATURES = 11

_DIRS = [[-1, 0], [0, 1], [1, 0], [0, -1]]  # (dy, dx)


def conquest_delta(cfg, att, dfn):
    """Change in a zone's capture progress over one decision, given the number of attackers
    and defenders inside it (Armagetron: rate x attackers - defend x defenders - decay, per s)."""
    d = (cfg.conquest_rate * att - cfg.defend_rate * dfn - cfg.conquest_decay) * cfg.decision_s
    # e.g. 1 vs 1 is exactly 0 (0.3 - 0.2 - 0.1) but lands at ~1e-9 in floats: snap it
    return d * (d.abs() > 1e-6)


class FortressEnv:
    def __init__(self, cfg: FortressConfig, num_envs: int, device="cpu", seed: int = 0):
        self.cfg = cfg
        self.N = num_envs
        self.T = cfg.team_size
        self.A = cfg.num_agents
        self.S = cfg.arena_size
        self.P = cfg.pad
        self.G = cfg.grid_size
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

        dev = self.device
        N, A, G, P, S = self.N, self.A, self.G, self.P, self.S
        self.dirs = torch.tensor(_DIRS, device=dev, dtype=torch.long)          # (4, 2)
        self.dir_lin = self.dirs[:, 0] * G + self.dirs[:, 1]                    # (4,)
        self.team = torch.arange(A, device=dev) // self.T                       # (A,)
        self.agent_ids = torch.arange(A, device=dev)

        # ---- static map: rim + zones ---------------------------------------
        base = torch.full((G, G), RIM, dtype=torch.int16, device=dev)
        base[P:P + S, P:P + S] = EMPTY
        self.base_owner = base

        r = cfg.zone_radius
        d = cfg.zone_center_offset  # integer centres keep the two zones exactly mirror-symmetric
        c = P + (S - 1) // 2
        far = P + S - 1
        # team 0 defends the bottom zone, team 1 the top zone (point-symmetric)
        self.zone_center = torch.tensor([[far - d, c], [P + d, c]], device=dev, dtype=torch.float32)
        yy, xx = torch.meshgrid(torch.arange(G, device=dev), torch.arange(G, device=dev), indexing="ij")
        zone = torch.full((G, G), -1, dtype=torch.int8, device=dev)
        for k in range(2):
            cy, cx = self.zone_center[k]
            inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
            inside &= base == EMPTY
            zone[inside] = k
        self.zone_grid = zone
        self.zone_flat = zone.view(-1).long()

        # ---- spawn layout ---------------------------------------------------
        # V (wingmen) formation around a point beside the zone centre, facing the enemy.
        # Team 0 faces up (-y), so 'back' is +y; slot sides alternate.
        T = self.T
        ly = far - d
        lx = c + int(round(cfg.spawn_side_m / cfg.cell_m))
        spawn = []
        for k in range(T):
            back, side, sign = cfg.wingmen(k)
            spawn.append([ly + back, lx + sign * side])
        for y, x in list(spawn):  # team 1 = 180 degree rotation of team 0
            spawn.append([2 * P + S - 1 - y, 2 * P + S - 1 - x])
        assert len({tuple(p) for p in spawn}) == len(spawn), "spawn cells overlap"
        self.spawn_pos = torch.tensor(spawn, device=dev, dtype=torch.long)     # (A, 2)
        self.spawn_dir = torch.tensor([0] * T + [2] * T, device=dev, dtype=torch.long)

        # ---- egocentric crop offsets ---------------------------------------
        R = cfg.obs_radius
        K = 2 * R + 1
        self.K = K
        i, j = torch.meshgrid(torch.arange(K, device=dev), torch.arange(K, device=dev), indexing="ij")
        fwd = (R - i)       # row 0 of the crop is furthest ahead
        right = (j - R)     # column 0 is furthest left
        offs = []
        for dd in range(4):
            f = self.dirs[dd]
            rt = self.dirs[(dd + 1) % 4]
            dy = fwd * f[0] + right * rt[0]
            dx = fwd * f[1] + right * rt[1]
            offs.append((dy * G + dx).reshape(-1))
        self.crop_off = torch.stack(offs)                                       # (4, K*K)

        # observation order of the other agents, per observer:
        # teammates (index order, excluding self) then enemies (index order)
        order = []
        for a in range(A):
            ta = a // T
            mates = [b for b in range(A) if b // T == ta and b != a]
            foes = [b for b in range(A) if b // T != ta]
            order.append(mates + foes)
        self.other_order = torch.tensor(order, device=dev, dtype=torch.long)    # (A, A-1)

        # grinding: speed units per tick from a wall k cells to the side (index 0 unused)
        self.accel_reach = cfg.accel_reach
        self.accel_by_dist = torch.tensor([0.0] + [cfg.accel_units(k) for k in range(1, self.accel_reach + 1)],
                                          device=dev, dtype=torch.float32)

        # explosion disk: every cell within explosion_radius of the crash point
        er = cfg.explosion_radius
        ri = int(er)
        blast = [dy * G + dx for dy in range(-ri, ri + 1) for dx in range(-ri, ri + 1)
                 if dy * dy + dx * dx <= er * er] if er > 0 else []
        self.blast_off = torch.tensor(blast, dtype=torch.long, device=dev)

        # self features (+ own slot one-hot if agent_id_obs), then the other agents
        self.n_pending = cfg.reaction_delay if cfg.pending_actions_obs else 0
        self.pending_off = SELF_FEATURES + (self.T if cfg.agent_id_obs else 0)
        self.self_dim = self.pending_off + 3 * self.n_pending
        self.vec_dim = self.self_dim + (A - 1) * OTHER_FEATURES
        self.ray_len = S

        # ---- dynamic state --------------------------------------------------
        self.owner = torch.empty((N, G, G), dtype=torch.int16, device=dev)
        self.stamp = torch.zeros((N, G, G), dtype=torch.int32, device=dev)
        self.pos = torch.zeros((N, A, 2), dtype=torch.long, device=dev)
        self.dir = torch.zeros((N, A), dtype=torch.long, device=dev)
        # speed and movement credit are float speed units (SPEED_UNIT credit = one cell)
        self.speed = torch.zeros((N, A), dtype=torch.float32, device=dev)
        self.credit = torch.zeros((N, A), dtype=torch.float32, device=dev)
        self.alive = torch.zeros((N, A), dtype=torch.bool, device=dev)
        self.death_tick = torch.zeros((N, A), dtype=torch.long, device=dev)
        self.death_cause = torch.full((N, A), CAUSE_ALIVE, dtype=torch.long, device=dev)
        self.progress = torch.zeros((N, 2), dtype=torch.float32, device=dev)
        self.tick = torch.zeros(N, dtype=torch.long, device=dev)
        self.steps = torch.zeros(N, dtype=torch.long, device=dev)
        self.near_wall = torch.zeros((N, A), dtype=torch.bool, device=dev)
        self.odo = torch.zeros((N, A), dtype=torch.long, device=dev)        # cells driven this round
        self.rubber = torch.zeros((N, A), dtype=torch.float32, device=dev)  # metres of rubber left
        self.rubber_used = torch.zeros((N, A), dtype=torch.float32, device=dev)  # metres burnt, last step
        self.pressing = torch.zeros((N, A), dtype=torch.bool, device=dev)  # held against a wall by rubber
        self.moved = torch.zeros((N, A), dtype=torch.long, device=dev)  # cells moved in the last step
        self.blasted = torch.zeros((N, A), dtype=torch.long, device=dev)  # enemy wall cells destroyed by this cycle's explosion, last step
        # breach tracking (measurement only, does not affect play). A "breach" is the set of
        # ENEMY wall cells destroyed by one cycle's explosion; it is identified by the id of
        # the cycle that exploded (each cycle dies at most once per round).
        self.breach_by = torch.full((N, G, G), -1, dtype=torch.int16, device=dev)
        self.breach_tick = torch.zeros((N, A), dtype=torch.long, device=dev)
        self.breach_made = torch.zeros((N, A), dtype=torch.bool, device=dev)
        self.breach_used = torch.zeros((N, A), dtype=torch.bool, device=dev)    # a teammate drove through it
        self.breach_passes = torch.zeros((N, A), dtype=torch.long, device=dev)  # teammate entries into it
        self.breach_enemy_passes = torch.zeros((N, A), dtype=torch.long, device=dev)
        # humanlike limits
        self.last_turn = torch.full((N, A), -10_000, dtype=torch.long, device=dev)
        self.turns_blocked = torch.zeros((N, A), dtype=torch.long, device=dev)  # last step
        L = cfg.reaction_delay + 1
        self.hist_codes = torch.zeros((L, N, A, self.K, self.K), dtype=torch.uint8, device=dev)
        self.hist_vec = torch.zeros((L, N, A, self.vec_dim), dtype=torch.float32, device=dev)
        self._action_hist = {}   # per-key ring buffers for delay_actions()
        # the policy's own last actions, most recent first (not yet visible in its delayed view)
        self.recent_actions = torch.zeros((max(1, self.n_pending), N, A), dtype=torch.long, device=dev)
        self.reset_done(torch.ones(N, dtype=torch.bool, device=dev))

    # =====================================================================
    # reset
    # =====================================================================
    def reset_done(self, mask: torch.Tensor):
        """Start a fresh round in every env where ``mask`` is True."""
        idx = mask.nonzero(as_tuple=True)[0]
        m = idx.numel()
        if m == 0:
            return
        cfg = self.cfg
        A, G = self.A, self.G
        self.owner[idx] = self.base_owner
        self.stamp[idx] = 0
        pos = self.spawn_pos.unsqueeze(0).repeat(m, 1, 1)
        if cfg.spawn_jitter > 0:
            j = torch.randint(-cfg.spawn_jitter, cfg.spawn_jitter + 1, (m, A),
                              generator=self.gen, device=self.device)
            pos[:, :, 1] += j
        self.pos[idx] = pos
        self.dir[idx] = self.spawn_dir
        self.speed[idx] = cfg.base_speed
        self.credit[idx] = 0
        self.alive[idx] = True
        self.death_tick[idx] = 0
        self.death_cause[idx] = CAUSE_ALIVE
        self.progress[idx] = 0.0
        self.tick[idx] = 0
        self.steps[idx] = 0
        self.near_wall[idx] = False
        self.moved[idx] = 0
        self.odo[idx] = 0
        self.rubber[idx] = cfg.rubber_m
        self.pressing[idx] = False
        self.rubber_used[idx] = 0.0
        self.breach_by[idx] = -1
        self.breach_tick[idx] = 0
        self.breach_made[idx] = False
        self.breach_used[idx] = False
        self.breach_passes[idx] = 0
        self.breach_enemy_passes[idx] = 0
        self.last_turn[idx] = -10_000
        self.recent_actions[:, idx] = 0
        self.turns_blocked[idx] = 0
        lin = pos[:, :, 0] * G + pos[:, :, 1]                                   # (m, A)
        flat = (idx[:, None] * G * G + lin).reshape(-1)
        self.owner.view(-1)[flat] = self.agent_ids.repeat(m).to(torch.int16)

    # =====================================================================
    # step
    # =====================================================================
    def step(self, actions: torch.Tensor) -> dict:
        """Advance every round by one decision. Does not reset finished rounds."""
        cfg = self.cfg
        alive_before = self.alive.clone()
        progress_before = self.progress.clone()
        cause_before = self.death_cause.clone()

        if self.n_pending:
            self.recent_actions = torch.roll(self.recent_actions, 1, dims=0)
            self.recent_actions[0] = actions.long()
        turn = torch.tensor([0, -1, 1], device=self.device)[actions.long()]
        if cfg.turn_cooldown > 0:
            allowed = (self.steps.unsqueeze(1) - self.last_turn) > cfg.turn_cooldown
            self.turns_blocked = (turn != 0) & ~allowed & self.alive
            turn = torch.where(allowed, turn, torch.zeros_like(turn))
            self.last_turn = torch.where((turn != 0) & self.alive, self.steps.unsqueeze(1), self.last_turn)
        self.turned = (turn != 0) & self.alive
        self.dir = torch.where(self.alive, (self.dir + turn) % 4, self.dir)

        self.moved.zero_()
        self.blasted.zero_()
        self.rubber_used.zero_()
        for _ in range(cfg.ticks_per_step):
            self._tick()
        self._clear_walls()
        self.steps += 1

        # ---- conquest ------------------------------------------------------
        zone_of = self.zone_flat[self.pos[:, :, 0] * self.G + self.pos[:, :, 1]]  # (N, A) -1/0/1
        zone_of = torch.where(self.alive, zone_of, torch.full_like(zone_of, -1))
        att = torch.zeros((self.N, 2), device=self.device)
        dfn = torch.zeros((self.N, 2), device=self.device)
        for k in range(2):
            in_k = zone_of == k
            att[:, k] = (in_k & (self.team != k)).sum(1).float()
            dfn[:, k] = (in_k & (self.team == k)).sum(1).float()
        self.progress = (self.progress + conquest_delta(cfg, att, dfn)).clamp(0.0, 1.0)

        # ---- round outcome ---------------------------------------------------
        team_alive = self.alive.view(self.N, 2, self.T).any(2)                  # (N, 2)
        conquered = self.progress >= 1.0 - 1e-5   # float sums can land just under 1
        lost = conquered | ~team_alive
        timeout = self.steps >= cfg.max_steps
        done = lost.any(1) | timeout
        winner = torch.full((self.N,), -1, dtype=torch.long, device=self.device)
        winner = torch.where(lost[:, 1] & ~lost[:, 0], torch.zeros_like(winner), winner)
        winner = torch.where(lost[:, 0] & ~lost[:, 1], torch.ones_like(winner), winner)
        reason = torch.full((self.N,), REASON_NONE, dtype=torch.long, device=self.device)
        reason = torch.where(timeout, torch.full_like(reason, REASON_TIMEOUT), reason)
        reason = torch.where((~team_alive).any(1), torch.full_like(reason, REASON_ELIMINATION), reason)
        reason = torch.where(conquered.any(1), torch.full_like(reason, REASON_CONQUEST), reason)

        died = alive_before & ~self.alive
        return {
            "alive_before": alive_before,
            "alive": self.alive.clone(),
            "died": died,
            "cause": torch.where(died, self.death_cause, torch.full_like(cause_before, CAUSE_ALIVE)),
            "progress_before": progress_before,
            "progress": self.progress.clone(),
            "zone_of": zone_of,              # which zone each alive cycle is in (-1 none)
            "attackers": att,                # (N, 2) enemies inside zone k
            "defenders": dfn,                # (N, 2) owners inside zone k
            "done": done,
            "winner": torch.where(done, winner, torch.full_like(winner, -1)),
            "reason": torch.where(done, reason, torch.zeros_like(reason)),
            "steps": self.steps.clone(),
            "turned": self.turned.clone(),
            "turns_blocked": self.turns_blocked.clone(),
            "blasted": self.blasted.clone(),  # enemy wall cells each dying cycle's explosion destroyed
            "rubber_used": self.rubber_used.clone(),  # metres of rubber burnt this step
            # cumulative for the round, per cycle that exploded:
            "breach_made": self.breach_made.clone(),
            "breach_used": self.breach_used.clone(),
            "breach_passes": self.breach_passes.clone(),
            "breach_enemy_passes": self.breach_enemy_passes.clone(),
        }

    def _tick(self):
        cfg = self.cfg
        N, A, G = self.N, self.A, self.G
        self.tick += 1
        owner_flat = self.owner.view(N, -1)

        self.credit = torch.where(self.alive, self.credit + self.speed, self.credit)
        # a cycle held against a wall tries to move again every tick
        move = self.alive & ((self.credit >= SPEED_UNIT) | self.pressing)
        self.credit = torch.where(move, (self.credit - SPEED_UNIT).clamp(min=0), self.credit)

        step = self.dirs[self.dir] * move.unsqueeze(-1).long()                  # (N, A, 2)
        new = self.pos + step
        lin = new[:, :, 0] * G + new[:, :, 1]
        target = owner_flat.gather(1, lin).long()
        hit = move & (target != EMPTY)

        same = (lin.unsqueeze(2) == lin.unsqueeze(1)) & move.unsqueeze(2) & move.unsqueeze(1)
        same &= ~torch.eye(A, dtype=torch.bool, device=self.device)
        headon = same.any(2)

        # rubber: a cycle blocked by a wall is held in place and burns the distance it would
        # have driven; it only dies once the rubber is gone. Head-on collisions are fatal.
        held = torch.zeros_like(hit)
        if cfg.rubber_m > 0:
            burn = self.speed * cfg.mps_per_unit * cfg.tick_s                        # metres this tick
            blocked = hit & ~headon
            held = blocked & (self.rubber - burn > 1e-6)
            self.rubber = torch.where(blocked, self.rubber - burn, self.rubber)
            self.rubber_used += torch.where(blocked, burn, torch.zeros_like(burn))
            # keep pressing: try again every tick while held, without banking movement credit
            self.credit = torch.where(held, torch.zeros_like(self.credit), self.credit)
            regen = cfg.rubber_m / cfg.rubber_time_s * cfg.tick_s
            self.rubber = torch.where(self.alive & ~blocked, (self.rubber + regen).clamp(max=cfg.rubber_m),
                                      self.rubber)
        self.pressing = held

        crash = (hit & ~held) | headon
        cause = torch.where(hit, target, torch.full_like(target, CAUSE_HEADON))
        self.alive &= ~crash
        self.death_tick = torch.where(crash, self.tick.unsqueeze(1).expand(N, A), self.death_tick)
        self.death_cause = torch.where(crash, cause, self.death_cause)

        ok = move & ~crash & ~held
        self.pos = torch.where(ok.unsqueeze(-1), new, self.pos)
        self.moved += ok.long()
        self.odo += ok.long()
        n_i, a_i = ok.nonzero(as_tuple=True)
        if n_i.numel():
            self._track_breach_passes(n_i, a_i, lin)
            flat = n_i * G * G + lin[n_i, a_i]
            self.owner.view(-1)[flat] = a_i.to(torch.int16)
            self.stamp.view(-1)[flat] = self.odo[n_i, a_i].to(torch.int32)

        if self.blast_off.numel():
            self._explode(crash, lin)

        # grinding: the nearest wall within reach on each side accelerates (rim only if accel_rim);
        # above base speed, speed decays by a fraction of the excess
        plin = self.pos[:, :, 0] * G + self.pos[:, :, 1]
        gain = torch.zeros_like(self.speed)
        for side in (3, 1):  # left, right
            dl = self.dir_lin[(self.dir + side) % 4]
            found = torch.zeros_like(self.alive)
            for k in range(1, self.accel_reach + 1):
                o = owner_flat.gather(1, (plin + k * dl).clamp(0, G * G - 1))
                wall = (o >= 0) | ((o == RIM) if cfg.accel_rim else torch.zeros_like(found))
                first = wall & ~found
                gain = gain + first.float() * self.accel_by_dist[k]
                found |= (o != EMPTY)            # anything in between blocks walls further out
        gain = gain * self.alive.float()
        excess = (self.speed - cfg.base_speed).clamp(min=0)
        decay = excess * cfg.decay_per_tick
        new_speed = (self.speed + gain - decay).clamp(cfg.base_speed, cfg.max_speed_units)
        self.speed = torch.where(self.alive, new_speed, self.speed)
        self.near_wall = gain > 0

    def _track_breach_passes(self, n_i, a_i, lin):
        """Movers (n_i, a_i) are entering cells lin[n_i, a_i]: count entries into fresh breaches."""
        G2 = self.G * self.G
        b = self.breach_by.view(-1)[n_i * G2 + lin[n_i, a_i]].long()
        has = b >= 0
        if not has.any():
            return
        n_b, a_b, b = n_i[has], a_i[has], b[has]
        fresh = self.tick[n_b] - self.breach_tick[n_b, b] <= self.cfg.breach_window_ticks
        n_b, a_b, b = n_b[fresh], a_b[fresh], b[fresh]
        mate = self.team[a_b] == self.team[b]
        one = torch.ones_like(b)
        self.breach_passes.index_put_((n_b[mate], b[mate]), one[mate], accumulate=True)
        self.breach_used[n_b[mate], b[mate]] = True
        self.breach_enemy_passes.index_put_((n_b[~mate], b[~mate]), one[~mate], accumulate=True)

    def _explode(self, crash, target_lin):
        """Every crash blows a hole: all trail cells within explosion_radius of the cell the
        cycle crashed into are removed. The rim is never broken and a living cycle's current
        cell is never removed."""
        n_i, a_i = crash.nonzero(as_tuple=True)
        if n_i.numel() == 0:
            return
        G2 = self.G * self.G
        cells = (n_i * G2 + target_lin[n_i, a_i]).unsqueeze(1) + self.blast_off.unsqueeze(0)  # (E, D)
        flat = self.owner.view(-1)
        o = flat[cells].long()
        trail = o >= 0
        enemy = trail & (self.team[o.clamp(min=0)] != self.team[a_i].unsqueeze(1))
        self.blasted.index_put_((n_i, a_i), enemy.sum(1), accumulate=True)
        # tag destroyed enemy wall cells as this cycle's breach
        e_rows, e_cols = enemy.nonzero(as_tuple=True)
        self.breach_by.view(-1)[cells[e_rows, e_cols]] = a_i[e_rows].to(torch.int16)
        made = enemy.any(1)
        self.breach_made[n_i, a_i] |= made
        self.breach_tick[n_i, a_i] = torch.where(made, self.tick[n_i], self.breach_tick[n_i, a_i])
        flat[cells[trail]] = EMPTY
        # restore the cells living cycles are sitting on
        ln, la = self.alive.nonzero(as_tuple=True)
        head = ln * G2 + self.pos[ln, la, 0] * self.G + self.pos[ln, la, 1]
        flat[head] = la.to(torch.int16)

    def _clear_walls(self):
        """Shorten walls to wall_length (distance driven since a cell was laid) and remove the
        walls of cycles that died more than walls_stay_up ago."""
        cfg = self.cfg
        N = self.N
        o = self.owner.long()
        trail = o >= 0
        oc = o.clamp(min=0).view(N, -1)
        clear = torch.zeros_like(trail)
        if cfg.wall_length_cells > 0:
            age = self.odo.gather(1, oc).view_as(trail) - self.stamp.long()
            clear |= trail & (age >= cfg.wall_length_cells)
        gone = ~self.alive & (self.tick.unsqueeze(1) - self.death_tick >= cfg.walls_stay_up_ticks)  # (N, A)
        clear |= trail & gone.gather(1, oc).view_as(trail)
        # never erase the cell a living cycle is sitting on
        n_i, a_i = self.alive.nonzero(as_tuple=True)
        clear.view(N, -1)[n_i, self.pos[n_i, a_i, 0] * self.G + self.pos[n_i, a_i, 1]] = False
        self.owner[clear] = EMPTY

    # =====================================================================
    # observation
    # =====================================================================
    def rays(self) -> torch.Tensor:
        """Free cells straight ahead / to the left / to the right of each head. (N, A, 3) long."""
        N, A, G, L = self.N, self.A, self.G, self.ray_len
        plin = self.pos[:, :, 0] * G + self.pos[:, :, 1]
        dirs = torch.stack([self.dir, (self.dir + 3) % 4, (self.dir + 1) % 4], dim=2)  # (N, A, 3)
        steps = torch.arange(1, L + 1, device=self.device)
        idx = plin[:, :, None, None] + self.dir_lin[dirs][..., None] * steps          # (N, A, 3, L)
        idx = idx.clamp(0, G * G - 1)
        cells = self.owner.view(N, -1).gather(1, idx.view(N, -1)).view(N, A, 3, L)
        return (cells == EMPTY).long().cumprod(-1).sum(-1)

    def _ego(self, dy, dx, d):
        """Rotate world offsets (dy, dx) into the frame of a cycle facing d -> (forward, right)."""
        f = self.dirs[d]
        r = self.dirs[(d + 1) % 4]
        fwd = dy * f[..., 0] + dx * f[..., 1]
        rgt = dy * r[..., 0] + dx * r[..., 1]
        return fwd, rgt

    def delay_actions(self, actions, key="bot"):
        """Humanlike reaction delay for controllers that read the game state directly
        (scripted bots): the action decided now is carried out ``reaction_delay`` decisions
        later; until then (start of a round) cycles go straight. Call once per step per key."""
        d = self.cfg.reaction_delay
        if d == 0:
            return actions
        L = d + 1
        buf = self._action_hist.get(key)
        if buf is None:
            buf = self._action_hist[key] = torch.zeros((L, self.N, self.A), dtype=torch.long,
                                                       device=self.device)
        n = torch.arange(self.N, device=self.device)
        # a turn decided in the last d steps hasn't happened yet: don't stack another one on it
        pending = torch.zeros_like(actions, dtype=torch.bool)
        for k in range(1, d + 1):
            past = buf[(self.steps - k) % L, n]
            pending |= (past != 0) & (self.steps >= k).unsqueeze(1)
        buf[self.steps % L, n] = torch.where(pending, torch.zeros_like(actions), actions.long())
        out = buf[(self.steps - d).clamp(min=0) % L, n]
        return torch.where((self.steps >= d).unsqueeze(1), out, torch.zeros_like(out))

    def observe(self):
        """
        Observation the policy acts on: the current view from ``reaction_delay`` decisions
        ago (or the round's first view if the round is younger than that). Safe to call
        more than once per step.
        """
        codes, vec = self.observe_now()
        d = self.cfg.reaction_delay
        if d == 0:
            return codes, vec
        codes, vec = self._delayed(codes, vec, d)
        if self.n_pending:
            oh = torch.nn.functional.one_hot(self.recent_actions[:self.n_pending], 3).float()  # (d, N, A, 3)
            vec = vec.clone()
            vec[..., self.pending_off:self.self_dim] = oh.permute(1, 2, 0, 3).reshape(self.N, self.A, -1)
        return codes, vec

    def _delayed(self, codes, vec, d):
        L = d + 1
        n = torch.arange(self.N, device=self.device)
        slot = self.steps % L
        self.hist_codes[slot, n] = codes
        self.hist_vec[slot, n] = vec
        src = (self.steps - d).clamp(min=0) % L
        return self.hist_codes[src, n], self.hist_vec[src, n]

    def observe_now(self):
        """
        Returns
        -------
        codes : (N, A, K, K) uint8   egocentric map, row 0 = straight ahead, col 0 = left
        vec   : (N, A, vec_dim) float32
        """
        cfg = self.cfg
        N, A, G, K, S = self.N, self.A, self.G, self.K, self.S
        dev = self.device
        plin = self.pos[:, :, 0] * G + self.pos[:, :, 1]                        # (N, A)

        # ---- egocentric crop -----------------------------------------------
        idx = plin.unsqueeze(-1) + self.crop_off[self.dir]                      # (N, A, KK)
        o = self.owner.view(N, -1).gather(1, idx.view(N, -1)).view(N, A, -1).long()
        heads = torch.full((N, G * G), -1, dtype=torch.long, device=dev)
        n_i, a_i = self.alive.nonzero(as_tuple=True)
        heads[n_i, plin[n_i, a_i]] = a_i
        h = heads.gather(1, idx.view(N, -1)).view(N, A, -1)
        z = self.zone_flat[idx]

        me = self.agent_ids.view(1, A, 1)
        my_team = self.team.view(1, A, 1)
        occ = torch.zeros_like(o)
        occ = torch.where(o == RIM, torch.full_like(occ, OCC_RIM), occ)
        o_team = self.team[o.clamp(min=0)]
        trail_code = torch.where(o == me, OCC_OWN, torch.where(o_team == my_team, OCC_TEAM, OCC_ENEMY))
        occ = torch.where(o >= 0, trail_code, occ)
        h_team = self.team[h.clamp(min=0)]
        head_code = torch.where(h_team == my_team, OCC_TEAM_HEAD, OCC_ENEMY_HEAD)
        occ = torch.where((h >= 0) & (h != me), head_code, occ)
        zc = torch.where(z < 0, ZONE_NONE, torch.where(z == my_team, ZONE_OWN, ZONE_ENEMY))
        codes = (occ * 3 + zc).to(torch.uint8).view(N, A, K, K)

        # ---- self features ---------------------------------------------------
        posf = self.pos.float()
        team_of = self.team.unsqueeze(0).expand(N, A)
        own_c = self.zone_center[team_of]                                        # (N, A, 2)
        foe_c = self.zone_center[1 - team_of]
        oz_f, oz_r = self._ego(own_c[..., 0] - posf[..., 0], own_c[..., 1] - posf[..., 1], self.dir)
        ez_f, ez_r = self._ego(foe_c[..., 0] - posf[..., 0], foe_c[..., 1] - posf[..., 1], self.dir)
        zone_of = self.zone_flat[plin]
        in_own = (zone_of == team_of).float()
        in_foe = (zone_of == 1 - team_of).float()
        prog_own = self.progress.gather(1, team_of)
        prog_foe = self.progress.gather(1, 1 - team_of)
        alive_team = self.alive.view(N, 2, self.T).float().mean(2)               # (N, 2)
        mates_alive = alive_team.gather(1, team_of)
        foes_alive = alive_team.gather(1, 1 - team_of)
        speed_n = (self.speed - cfg.base_speed).float() / max(1, cfg.max_speed_units - cfg.base_speed)
        rays = self.rays().float() / S
        t_frac = (self.steps.float() / cfg.max_steps).unsqueeze(1).expand(N, A)
        self_feat = torch.stack([
            self.alive.float(), speed_n, t_frac,
            oz_f / S, oz_r / S, ez_f / S, ez_r / S, in_own, in_foe,
            prog_own, prog_foe, mates_alive, foes_alive,
            rays[..., 0], rays[..., 1], rays[..., 2],
            self.rubber / cfg.rubber_m if cfg.rubber_m > 0 else torch.ones_like(speed_n),
        ], dim=-1)                                                                # (N, A, 17)

        # ---- other agents (pairwise, observer i -> other j) --------------------
        dy = posf[:, None, :, 0] - posf[:, :, None, 0]                          # (N, i, j)
        dx = posf[:, None, :, 1] - posf[:, :, None, 1]
        di = self.dir.unsqueeze(2).expand(N, A, A)
        rf, rr = self._ego(dy, dx, di)
        rel_dir = (self.dir.unsqueeze(1) - self.dir.unsqueeze(2)) % 4            # (N, i, j)
        rel_dir_oh = torch.nn.functional.one_hot(rel_dir, 4).float()
        alive_j = self.alive.unsqueeze(1).expand(N, A, A).float()
        visible = self.alive.unsqueeze(1).expand(N, A, A)
        if cfg.vis_radius > 0:
            same_team = (self.team.view(1, A, 1) == self.team.view(1, 1, A))
            close = dy * dy + dx * dx <= cfg.vis_radius ** 2
            visible = visible & (close | same_team)
        vis = visible.float()
        z_j = zone_of.unsqueeze(1).expand(N, A, A)
        t_i = team_of.unsqueeze(2).expand(N, A, A)
        j_in_my = (z_j == t_i).float()
        j_in_foe = (z_j == 1 - t_i).float()
        speed_j = speed_n.unsqueeze(1).expand(N, A, A)
        pair = torch.cat([
            alive_j.unsqueeze(-1), vis.unsqueeze(-1),
            (rf / S).unsqueeze(-1), (rr / S).unsqueeze(-1),
            rel_dir_oh,
            speed_j.unsqueeze(-1), j_in_my.unsqueeze(-1), j_in_foe.unsqueeze(-1),
        ], dim=-1)                                                                # (N, i, j, 11)
        pair[..., 2:] = pair[..., 2:] * vis.unsqueeze(-1)
        order = self.other_order.view(1, A, A - 1, 1).expand(N, A, A - 1, OTHER_FEATURES)
        others = pair.gather(2, order).reshape(N, A, -1)

        parts = [self_feat]
        if cfg.agent_id_obs:
            slot = torch.nn.functional.one_hot(self.agent_ids % self.T, self.T).float()   # (A, T)
            parts.append(slot.unsqueeze(0).expand(N, A, self.T))
        if self.n_pending:  # filled in by observe() with the current values
            parts.append(torch.zeros((N, A, 3 * self.n_pending), device=dev))
        parts.append(others)
        vec = torch.cat(parts, dim=-1)
        return codes, vec

    # =====================================================================
    # helpers for tests / rendering
    # =====================================================================
    def check_invariants(self) -> list:
        """Cheap consistency checks on the whole batch. Returns a list of problems (empty = OK).
        The trainer runs this periodically so a silent simulator bug stops the run loudly."""
        errs = []
        N, A, G, P, S, cfg = self.N, self.A, self.G, self.P, self.S, self.cfg
        o = self.owner
        if (o < RIM).any() or (o >= A).any():
            errs.append("owner grid holds an invalid id")
        rim = torch.ones((G, G), dtype=torch.bool, device=self.device)
        rim[P:P + S, P:P + S] = False
        if (o[:, rim] != RIM).any():
            errs.append("rim band was modified")
        if (o[:, ~rim] == RIM).any():
            errs.append("rim value inside the arena")
        n_i, a_i = self.alive.nonzero(as_tuple=True)
        y, x = self.pos[n_i, a_i, 0], self.pos[n_i, a_i, 1]
        if ((y < P) | (y >= P + S) | (x < P) | (x >= P + S)).any():
            errs.append("living cycle outside the arena")
        elif (o[n_i, y, x].long() != a_i).any():
            errs.append("living cycle is not on a cell it owns")
        lin = self.pos[:, :, 0] * G + self.pos[:, :, 1]
        both = self.alive.unsqueeze(2) & self.alive.unsqueeze(1)
        both &= ~torch.eye(A, dtype=torch.bool, device=self.device)
        if ((lin.unsqueeze(2) == lin.unsqueeze(1)) & both).any():
            errs.append("two living cycles on the same cell")
        live_speed = self.speed[self.alive]
        if ((live_speed < cfg.base_speed) | (live_speed > cfg.max_speed_units)).any():
            errs.append("speed out of range")
        if ((self.progress < 0) | (self.progress > 1)).any():
            errs.append("conquest progress out of [0, 1]")
        if (self.death_cause[self.alive] != CAUSE_ALIVE).any():
            errs.append("living cycle has a death cause")
        if (self.breach_by < -1).any() or (self.breach_by >= A).any():
            errs.append("breach grid holds an invalid id")
        return errs

    def clear_board(self, n: int = 0):
        """Empty env n completely (only the rim remains); all cycles dead and removed."""
        self.owner[n] = self.base_owner
        self.stamp[n] = 0
        self.alive[n] = False
        self.death_cause[n] = CAUSE_ALIVE
        self.progress[n] = 0.0
        self.steps[n] = 0
        self.tick[n] = 0

    def place(self, n: int, agent: int, y: int, x: int, d: int, speed=None, interior=True):
        """Put a living cycle at arena cell (y, x) (interior coordinates unless interior=False)."""
        if interior:
            y, x = y + self.P, x + self.P
        self.pos[n, agent, 0] = y
        self.pos[n, agent, 1] = x
        self.dir[n, agent] = d
        self.speed[n, agent] = float(self.cfg.base_speed if speed is None else speed)
        self.credit[n, agent] = 0
        self.alive[n, agent] = True
        self.death_cause[n, agent] = CAUSE_ALIVE
        self.owner[n, y, x] = agent
        self.stamp[n, y, x] = int(self.odo[n, agent])
        self.rubber[n, agent] = self.cfg.rubber_m
        self.pressing[n, agent] = False

    def set_cell(self, n: int, y: int, x: int, value: int, interior=True):
        if interior:
            y, x = y + self.P, x + self.P
        self.owner[n, y, x] = value
        self.stamp[n, y, x] = int(self.odo[n, value]) if value >= 0 else 0

    def cell(self, n: int, y: int, x: int, interior=True) -> int:
        if interior:
            y, x = y + self.P, x + self.P
        return int(self.owner[n, y, x])

    def interior_pos(self, n: int, agent: int):
        return int(self.pos[n, agent, 0]) - self.P, int(self.pos[n, agent, 1]) - self.P
