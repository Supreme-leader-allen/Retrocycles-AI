"""
Game configuration for the Fortress simulator, in real-world units.

Every default is taken from Armagetron Advanced (which Retrocycles is based on):
the "CVS Test Server: Fortress" configuration shipped with the game
(config/examples/fortress_soccer.cfg + cvs_test/fortress_physics.cfg), its map
Z-Man/fortress/for_old_clients-0.1.0.aamap.xml, and the engine's built-in
defaults (src/tron/gCycleMovement.cpp, gCycle.cpp, gArena.cpp) where the
server config does not override them. Measured in-game: end to end of the map
16.55 s and zone to zone 10.35 s, consistent with 500 m at 30 m/s and
320 m (400 m between zone centres minus two 40 m radii).

Scale
-----
1 grid cell = ``cell_m`` = 3 m and 1 decision = ``decision_s`` = 0.1 s, so the
base speed of 30 m/s is exactly one cell per decision. Each decision is split
into ``ticks_per_step`` physics ticks; a cycle moves at most one cell per tick
(max 90 m/s). Speeds are stored as integers: ``SPEED_UNIT`` credit = one cell.

Rules modelled
--------------
- Cycles drive forward, turn 90 degrees left/right (4 arena axes), at most once
  per decision (CYCLE_DELAY 0.1 s = one decision).
- Walls are ``wall_length_m`` long (measured in distance driven, like
  Armagetron) and a dead cycle's walls stay up ``walls_stay_up_s`` seconds.
- Rubber: a cycle trying to drive into a wall is held in front of it and burns
  rubber equal to the distance it would have travelled; it dies when the rubber
  is gone. Rubber refills over ``rubber_time_s``. Head-on collisions kill at once.
- Grinding: walls within ``wall_near_m`` on the left/right accelerate the cycle
  (Armagetron's 1/distance formula). The rim does not accelerate (CYCLE_ACCEL_RIM 0).
  Above base speed, speed decays by ``speed_decay_above`` x (excess) per second.
- Every crash explodes: trail walls (any team, never the rim) within
  ``explosion_radius_m`` of the crash point are destroyed.
- Fortress: per second, a zone's capture progress changes by
  conquest_rate x attackers - defend_rate x defenders - conquest_decay
  (wiki: 1 attacker alone 5 s, 1 vs 1 never, 2 vs 0 2 s, 2 vs 1 3.3 s, 2 vs 2 10 s).
  A team loses when its zone is captured or all its cycles are dead.
- Spawn: each team's cycles start in a V ("wingmen") formation around a point
  beside their own zone's centre, facing the enemy zone.

Humanlike limits
----------------
- ``reaction_delay``: the policy acts on the view from that many decisions ago.
- ``turn_cooldown``: extra decisions to wait between turns (0 = CYCLE_DELAY only).

Not modelled: brakes, continuous geometry (walls are 3 m grid cells), rubber
slowing the cycle down before contact, more than two teams.
"""
import math
from dataclasses import dataclass, asdict

SPEED_UNIT = 3000  # movement credit needed to advance one cell


@dataclass
class FortressConfig:
    team_size: int = 7

    # ---- scale ---------------------------------------------------------------
    cell_m: float = 3.0               # metres per grid cell
    decision_s: float = 0.1           # seconds per decision (= CYCLE_DELAY)
    ticks_per_step: int = 3           # physics ticks per decision

    # ---- map (for_old_clients-0.1.0.aamap.xml) --------------------------------
    arena_m: float = 500.0            # square arena side
    zone_radius_m: float = 40.0       # WIN_ZONE_INITIAL_SIZE 40
    zone_offset_m: float = 50.0       # zone centre to its rim
    spawn_side_m: float = 5.0         # spawn point is 5 m beside the zone centre
    wingmen_back_m: float = 2.202896  # SPAWN_WINGMEN_BACK
    wingmen_side_m: float = 2.75362   # SPAWN_WINGMEN_SIDE
    spawn_jitter: int = 0             # random +- cells added to spawn x (the real game has none)

    # ---- cycle physics --------------------------------------------------------
    cycle_speed: float = 30.0         # CYCLE_SPEED (m/s), fortress server
    max_speed: float = 90.0           # grid limit (one cell per tick); Armagetron has no cap
    cycle_accel: float = 20.0         # CYCLE_ACCEL, fortress server
    accel_offset_m: float = 2.0       # CYCLE_ACCEL_OFFSET
    wall_near_m: float = 6.0          # CYCLE_WALL_NEAR
    accel_rim: bool = False           # CYCLE_ACCEL_RIM 0: the rim gives no boost
    speed_decay_above: float = 0.1    # CYCLE_SPEED_DECAY_ABOVE (fraction of excess speed lost per s)
    rubber_m: float = 5.0             # CYCLE_RUBBER, fortress server; 0 = die on contact
    rubber_time_s: float = 10.0       # CYCLE_RUBBER_TIME: full refill time
    wall_length_m: float = 400.0      # WALLS_LENGTH, fortress server; 0 = infinite
    walls_stay_up_s: float = 8.0      # CYCLE_WALLS_STAY_UP_DELAY
    explosion_radius_m: float = 4.0   # EXPLOSION_RADIUS engine default (the CVS server uses 2);
                                      # 0 = no explosions

    # ---- fortress zone (fortress_soccer.cfg), per second -----------------------
    conquest_rate: float = 0.3        # FORTRESS_CONQUEST_RATE, per attacker
    defend_rate: float = 0.2          # FORTRESS_DEFEND_RATE, per defender
    conquest_decay: float = 0.1       # FORTRESS_CONQUEST_DECAY_RATE, always applied

    # ---- round -----------------------------------------------------------------
    max_time_s: float = 120.0         # round length limit (draw)
    breach_window_s: float = 3.0      # measurement only: a hole counts as "used" if a cycle drives
                                      # into it within this many seconds of the explosion

    # ---- humanlike limits --------------------------------------------------------
    reaction_delay: int = 2           # decisions (0.2 s)
    turn_cooldown: int = 0

    # ---- observation ---------------------------------------------------------------
    agent_id_obs: bool = True         # each cycle sees its own slot number (one-hot)
    obs_radius: int = 10              # egocentric crop is (2r+1)^2 cells (r = 30 m)
    vis_radius: float = 0.0           # >0: enemies further than this many cells are hidden

    def __post_init__(self):
        assert self.team_size >= 1
        assert self.arena_size % 2 == 1, "arena must be an odd number of cells (exact symmetry)"
        assert 0 <= self.explosion_radius < self.pad, "explosion must fit inside the rim padding"
        assert self.reaction_delay >= 0 and self.turn_cooldown >= 0
        assert 0 < self.base_speed <= self.max_speed_units <= SPEED_UNIT, "max one cell per tick"
        assert self.zone_center_offset > self.zone_radius, "zone must not overlap the rim"
        back = max(self.wingmen(k)[0] for k in range(self.team_size))
        assert self.zone_center_offset - back >= 1, "spawn formation reaches the rim"

    # ---- derived (grid units) --------------------------------------------------
    @property
    def tick_s(self) -> float:
        return self.decision_s / self.ticks_per_step

    @property
    def arena_size(self) -> int:
        n = int(round(self.arena_m / self.cell_m))
        return n if n % 2 == 1 else n + 1

    @property
    def zone_radius(self) -> float:
        return self.zone_radius_m / self.cell_m

    @property
    def zone_center_offset(self) -> int:
        # cell index of the zone centre counted from its rim (cell i spans [i, i+1) * cell_m)
        return int(round(self.zone_offset_m / self.cell_m - 0.5))

    def wingmen(self, k: int):
        """(cells back, cells to the side, side sign) of team slot k relative to the spawn point
        (Armagetron gSpawnPoint::FindPos: slots alternate sides, each pair one step further out)."""
        if k == 0:
            return 0, 0, 0
        away = (k + 1) // 2
        sign = 1 if k % 2 == 1 else -1
        back = max(1, int(round(away * self.wingmen_back_m / self.cell_m)))
        side = max(1, int(round(away * self.wingmen_side_m / self.cell_m)))
        return back, side, sign

    @property
    def mps_per_unit(self) -> float:
        """m/s represented by one speed unit (cells/tick * SPEED_UNIT)."""
        return self.cell_m / self.tick_s / SPEED_UNIT

    @property
    def base_speed(self) -> int:
        return int(round(self.cycle_speed / self.mps_per_unit))

    @property
    def max_speed_units(self) -> int:
        return int(round(self.max_speed / self.mps_per_unit))

    def accel_units(self, k: int) -> float:
        """Speed units gained per tick from one wall k cells to the side (wall surface at
        (k - 0.5) cells), using Armagetron's acceleration falloff with distance."""
        d = (k - 0.5) * self.cell_m
        if d >= self.wall_near_m:
            return 0
        off, near = self.accel_offset_m, self.wall_near_m
        f = (1 / (d + off) - 1 / (near + off)) / (1 / off - 1 / (near + off))
        return self.cycle_accel * f * self.tick_s / self.mps_per_unit

    @property
    def accel_reach(self) -> int:
        """How many cells to the side can still accelerate."""
        k = 1
        while self.accel_units(k + 1) > 0:
            k += 1
        return k

    @property
    def decay_per_tick(self) -> float:
        return self.speed_decay_above * self.tick_s

    @property
    def wall_length_cells(self) -> int:
        return int(round(self.wall_length_m / self.cell_m)) if self.wall_length_m > 0 else 0

    @property
    def walls_stay_up_ticks(self) -> int:
        return int(round(self.walls_stay_up_s / self.tick_s))

    @property
    def explosion_radius(self) -> float:
        return self.explosion_radius_m / self.cell_m

    @property
    def max_steps(self) -> int:
        return int(round(self.max_time_s / self.decision_s))

    @property
    def breach_window_ticks(self) -> int:
        return int(round(self.breach_window_s / self.tick_s))

    @property
    def num_agents(self) -> int:
        return 2 * self.team_size

    @property
    def pad(self) -> int:
        # rim thickness around the interior; >= obs_radius so crops never go out of bounds
        return self.obs_radius + 1

    @property
    def grid_size(self) -> int:
        return self.arena_size + 2 * self.pad

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Rebuild a config saved in a checkpoint. Settings this version doesn't know (from older
        code) are dropped; settings added since get today's defaults, except agent_id_obs, which
        older checkpoints were trained without."""
        d = dict(d)
        d.setdefault("agent_id_obs", False)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
