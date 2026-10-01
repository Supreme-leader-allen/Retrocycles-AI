"""
Game configuration for the Fortress simulator.

Units
-----
- Distance: grid cells. The arena is an S x S square of cells.
- Time: ticks. One agent decision ("step") = ``ticks_per_step`` ticks.
- Speed: hundredths of a cell per tick (integers, so movement is exact and
  never depends on float rounding). A cycle moves one cell every time its
  movement credit reaches 100. At ``base_speed=50`` and 2 ticks/step a cycle
  moves exactly 1 cell per decision; at ``max_speed=100`` it moves 2.

Rules modelled (Retrocycles / Armagetron "Fortress", simplified to a grid):
- Cycles drive forward continuously, can turn 90 degrees left/right once per
  decision, and die on entering any occupied cell (rim, any trail, any head).
- Riding alongside a wall (an occupied cell directly to the left or right of
  the head) accelerates the cycle up to ``max_speed``; away from walls it
  decays back to ``base_speed``. This is Armagetron's "grinding" mechanic.
- Every crash explodes: all trail cells (any team) within ``explosion_radius``
  of the cell the cycle crashed into are destroyed, which can open a hole in
  the wall that was hit. The rim is indestructible.
- Trails are finite: a trail cell disappears ``trail_ticks`` after it was laid.
  A dead cycle's whole trail disappears ``dead_wall_ticks`` after it died.
- Each team owns a circular base zone. Enemy cycles inside it build conquest
  progress, defenders inside it reduce it, and it decays when no enemy is in
  it. A team loses when its zone reaches progress 1.0 or when all of its
  cycles are dead. Hitting ``max_steps`` is a draw.

Humanlike limits:
- ``reaction_delay``: the observation a policy acts on is from that many
  decisions ago (a human's reaction time). The simulation itself is not delayed.
- ``turn_cooldown``: minimum decisions between two turns (0 = can turn every decision).

Not modelled (deliberately): rubber, brakes, continuous turning geometry,
more than two teams. These are listed in README.md under "Simplifications".
"""
from dataclasses import dataclass, asdict

SPEED_UNIT = 100  # movement credit needed to advance one cell


@dataclass
class FortressConfig:
    team_size: int = 7
    arena_size: int = 0          # interior side length in cells; 0 = auto (29 + 6*team_size, odd)
    ticks_per_step: int = 2

    base_speed: int = 50         # 1/100 cell per tick
    max_speed: int = 100
    wall_accel: int = 4          # speed gained per tick while riding next to a wall
    speed_decay: int = 2         # speed lost per tick when not next to a wall

    trail_ticks: int = 200       # lifetime of a trail cell; 0 = trails never expire
    dead_wall_ticks: int = 30    # a dead cycle's trail vanishes this long after death
    explosion_radius: float = 2.0  # every crash removes all trail cells within this many cells
                                   # of the crash point (never the rim); 0 = no explosions

    zone_radius: float = 0.0     # 0 = auto (12% of arena, min 3)
    conquest_rate: float = 0.015  # progress per step per attacker in the zone
    defend_rate: float = 0.02     # progress removed per step per defender in the zone
    conquest_decay: float = 0.01  # progress removed per step when no attacker is in the zone

    breach_window_ticks: int = 60  # measurement only: a hole counts as "used" if a cycle drives
                                   # into it within this many ticks of the explosion
    max_steps: int = 500         # decisions per round before it is called a draw

    # humanlike limits
    reaction_delay: int = 2      # policy sees the game as it was this many decisions ago (0 = instant)
    turn_cooldown: int = 0       # after a turn, further turns are ignored for this many decisions

    obs_radius: int = 10         # egocentric crop is (2r+1) x (2r+1)
    vis_radius: float = 0.0      # >0: cycles further than this are hidden in the vector obs
    spawn_jitter: int = 1        # random +-cells added to each spawn x position

    def __post_init__(self):
        if self.arena_size <= 0:
            self.arena_size = 29 + 6 * self.team_size
        if self.zone_radius <= 0:
            self.zone_radius = max(3.0, round(0.12 * self.arena_size, 1))
        assert self.team_size >= 1
        assert self.arena_size % 2 == 1, "arena_size must be odd so the map is exactly symmetric"
        assert 0 <= self.explosion_radius < self.pad, "explosion must fit inside the rim padding"
        assert self.reaction_delay >= 0 and self.turn_cooldown >= 0
        assert self.base_speed > 0 and self.max_speed >= self.base_speed
        assert self.max_speed * self.ticks_per_step <= 4 * SPEED_UNIT, "too fast for the grid"
        spacing = self.arena_size / (self.team_size + 1)
        assert spacing >= 2 * self.spawn_jitter + 2, "arena too small for this team size / jitter"
        # zone centre sits zone_radius+3 from its rim; spawn row is zone_radius+2 further in
        assert 2 * (2 * self.zone_radius + 5) + 4 < self.arena_size, "zones/spawn rows overlap"

    # ---- derived -----------------------------------------------------------
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
