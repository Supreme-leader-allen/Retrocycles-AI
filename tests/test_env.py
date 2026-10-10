"""
Hand-constructed game states with known expected outcomes, checked by running
the simulator (not by reading it). Run:  python -m pytest -q
"""
import torch
import pytest

from tron.config import FortressConfig
from tron.env import (FortressEnv, EMPTY, RIM, CAUSE_RIM, CAUSE_HEADON, CAUSE_ALIVE,
                      REASON_CONQUEST, REASON_ELIMINATION, REASON_TIMEOUT,
                      OCC_ENEMY, OCC_OWN, OCC_TEAM, OCC_ENEMY_HEAD, OCC_RIM, ZONE_NONE)

S_, L_, R_ = 0, 1, 2  # straight, left, right
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


def make(team_size=1, n=1, **kw):
    # rule tests isolate one mechanic at a time; each of these is tested separately below
    kw.setdefault("wall_length_m", 0)          # infinite walls
    kw.setdefault("walls_stay_up_s", 1000.0)   # dead walls stay
    kw.setdefault("explosion_radius_m", 0)
    kw.setdefault("rubber_m", 0)               # die on contact
    kw.setdefault("reaction_delay", 0)
    cfg = FortressConfig(team_size=team_size, **kw)
    return FortressEnv(cfg, num_envs=n)


def act(env, *per_agent):
    """Same action list for every env."""
    a = torch.tensor(per_agent, dtype=torch.long).unsqueeze(0).repeat(env.N, 1)
    return env.step(a)


def blank(env, n=0):
    env.clear_board(n)


# ---------------------------------------------------------------- movement
def test_straight_move_leaves_trail():
    env = make()
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 30, 30, LEFT)
    for _ in range(3):
        act(env, S_, S_)
    assert env.interior_pos(0, 0) == (10, 13)
    for x in range(10, 14):
        assert env.cell(0, 10, x) == 0
    assert env.cell(0, 10, 14) == EMPTY


@pytest.mark.parametrize("start,action,expected_dir,expected_pos", [
    (UP, L_, LEFT, (10, 9)),
    (UP, R_, RIGHT, (10, 11)),
    (RIGHT, L_, UP, (9, 10)),
    (RIGHT, R_, DOWN, (11, 10)),
    (DOWN, L_, RIGHT, (10, 11)),
    (LEFT, R_, UP, (9, 10)),
])
def test_turns(start, action, expected_dir, expected_pos):
    env = make()
    blank(env)
    env.place(0, 0, 10, 10, start)
    env.place(0, 1, 30, 30, LEFT)
    act(env, action, S_)
    assert int(env.dir[0, 0]) == expected_dir
    assert env.interior_pos(0, 0) == expected_pos


def test_rim_kills():
    env = make()
    blank(env)
    env.place(0, 0, 0, 5, UP)
    env.place(0, 1, 30, 30, LEFT)
    info = act(env, S_, S_)
    assert not env.alive[0, 0]
    assert int(info["cause"][0, 0]) == CAUSE_RIM
    assert info["died"][0, 0] and not info["died"][0, 1]
    assert env.interior_pos(0, 0) == (0, 5)


def test_enemy_trail_kills_and_cause_is_owner():
    env = make()
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 30, 30, LEFT)
    env.set_cell(0, 10, 12, 1)
    info = act(env, S_, S_)
    assert env.alive[0, 0]
    info = act(env, S_, S_)
    assert not env.alive[0, 0]
    assert int(info["cause"][0, 0]) == 1
    assert env.interior_pos(0, 0) == (10, 11)


def test_teammate_trail_is_friendly_fire():
    env = make(team_size=2)
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 20, 20, UP)      # teammate
    env.place(0, 2, 30, 30, LEFT)
    env.set_cell(0, 10, 11, 1)
    info = act(env, S_, S_, S_, S_)
    assert int(info["cause"][0, 0]) == 1
    assert int(env.team[1]) == int(env.team[0])


def test_head_on_into_same_cell_kills_both():
    env = make()
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 10, 12, LEFT)
    info = act(env, S_, S_)
    assert not env.alive[0].any()
    assert info["cause"][0].tolist() == [CAUSE_HEADON, CAUSE_HEADON]
    assert env.cell(0, 10, 11) == EMPTY


def test_swap_kills_both():
    env = make()
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 10, 11, LEFT)
    info = act(env, S_, S_)
    assert info["cause"][0].tolist() == [1, 0]


def test_cell_vacated_this_tick_is_still_a_wall():
    env = make()
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 10, 11, UP)
    info = act(env, S_, S_)
    assert not env.alive[0, 0] and env.alive[0, 1]
    assert int(info["cause"][0, 0]) == 1


def test_crashed_cycle_trail_cell_does_not_move():
    env = make()
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 30, 30, LEFT)
    env.set_cell(0, 10, 11, RIM)
    act(env, S_, S_)
    assert env.cell(0, 10, 10) == 0 and env.cell(0, 10, 11) == RIM


# ---------------------------------------------------------------- speed
def test_no_acceleration_in_open_space():
    env = make()
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 100, 30, LEFT)
    for _ in range(5):
        act(env, S_, S_)
    assert int(env.speed[0, 0]) == env.cfg.base_speed
    assert env.interior_pos(0, 0) == (10, 15)


def _wall_line(env, y, x0, x1, owner=1):
    for x in range(x0, x1):
        env.set_cell(0, y, x, owner)


def test_rim_gives_no_boost():
    env = make()
    blank(env)
    env.place(0, 0, 0, 1, RIGHT)          # rim directly above
    env.place(0, 1, 30, 30, UP)
    for _ in range(10):
        act(env, S_, S_)
    assert int(env.speed[0, 0]) == env.cfg.base_speed
    assert env.interior_pos(0, 0) == (0, 11)


def test_grinding_a_wall_accelerates_and_leaves_no_gaps():
    env = make()
    blank(env)
    _wall_line(env, 9, 0, 160)            # enemy wall directly above row 10
    env.place(0, 0, 10, 1, RIGHT)
    env.place(0, 1, 100, 30, UP)
    for _ in range(60):                   # 6 s of grinding: v = 30 + 90(1 - e^-0.6) ~ 70 m/s
        act(env, S_, S_)
    assert int(env.speed[0, 0]) > 2 * env.cfg.base_speed
    assert int(env.moved[0, 0]) >= 2
    y, x = env.interior_pos(0, 0)
    for xx in range(1, x + 1):
        assert env.cell(0, 10, xx) == 0, f"gap in trail at x={xx}"


def test_grinding_strength_falls_off_with_distance():
    gains = []
    for gap in (1, 2, 3):
        env = make()
        blank(env)
        _wall_line(env, 10 - gap, 0, 40)
        env.place(0, 0, 10, 1, RIGHT)
        env.place(0, 1, 100, 30, UP)
        act(env, S_, S_)
        gains.append(float(env.speed[0, 0]) - env.cfg.base_speed)
    assert gains[0] > gains[1] > 0 and gains[2] == 0
    cfg = make().cfg
    assert gains[0] == pytest.approx(cfg.ticks_per_step * cfg.accel_units(1), rel=0.01)  # minus tiny decay


def test_walls_on_both_sides_add_up():
    env = make()
    blank(env)
    _wall_line(env, 9, 0, 40)
    _wall_line(env, 11, 0, 40)
    env.place(0, 0, 10, 1, RIGHT)
    env.place(0, 1, 100, 30, UP)
    act(env, S_, S_)
    gain = float(env.speed[0, 0]) - env.cfg.base_speed
    assert gain == pytest.approx(2 * env.cfg.ticks_per_step * env.cfg.accel_units(1), rel=0.01)


def test_boost_decays_slowly_after_leaving_the_wall():
    env = make()
    blank(env)
    fast = 2 * env.cfg.base_speed                 # 60 m/s
    env.place(0, 0, 10, 10, RIGHT, speed=fast)
    env.place(0, 1, 100, 30, LEFT)
    for _ in range(10):                           # 1 s in the open
        act(env, S_, S_)
    lost = fast - int(env.speed[0, 0])
    excess = fast - env.cfg.base_speed
    # ~10% of the excess per second (CYCLE_SPEED_DECAY_ABOVE 0.1), allowing for per-tick rounding up
    assert 0.08 * excess <= lost <= 0.12 * excess


def test_fast_cycle_dies_at_the_wall_on_the_last_free_cell():
    env = make()
    blank(env)
    top = env.cfg.max_speed_units                 # 90 m/s: moves on tick 1, then slightly slower
    env.place(0, 0, 10, 10, RIGHT, speed=top)
    env.place(0, 1, 100, 30, LEFT)
    env.set_cell(0, 10, 12, 1)
    act(env, S_, S_)
    assert not env.alive[0, 0]
    assert env.interior_pos(0, 0) == (10, 11)

    env2 = make()
    blank(env2)
    env2.place(0, 0, 10, 10, RIGHT, speed=top)
    env2.place(0, 1, 100, 30, LEFT)
    env2.set_cell(0, 10, 11, 1)
    act(env2, S_, S_)
    assert not env2.alive[0, 0] and env2.interior_pos(0, 0) == (10, 10)


# ---------------------------------------------------------------- rubber
def test_rubber_saves_a_cycle_that_turns_in_time():
    env = make(rubber_m=5.0)
    blank(env)
    _wall_line(env, 10, 11, 12)                   # single wall cell ahead
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 100, 30, LEFT)
    info = act(env, S_, S_)                       # blocked on tick 3: held, burns 1 m
    assert env.alive[0, 0] and env.interior_pos(0, 0) == (10, 10)
    assert float(info["rubber_used"][0, 0]) == pytest.approx(1.0)
    act(env, L_, S_)                              # turns away and escapes
    assert env.alive[0, 0] and env.interior_pos(0, 0) == (9, 10)


def test_rubber_runs_out_after_five_metres():
    env = make(rubber_m=5.0)
    blank(env)
    _wall_line(env, 10, 11, 12)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 100, 30, LEFT)
    act(env, S_, S_)                              # tick 3: 4 m left
    act(env, S_, S_)                              # ticks 4-6: 1 m left
    assert env.alive[0, 0]
    info = act(env, S_, S_)                       # tick 7: gone
    assert not env.alive[0, 0] and int(info["cause"][0, 0]) == 1
    assert env.interior_pos(0, 0) == (10, 10)


def test_faster_cycles_burn_rubber_faster():
    env = make(rubber_m=5.0)
    blank(env)
    _wall_line(env, 10, 11, 12)
    env.place(0, 0, 10, 10, RIGHT, speed=2 * env.cfg.base_speed)
    env.place(0, 1, 100, 30, LEFT)
    info = act(env, S_, S_)
    # 60 m/s: first blocked on tick 2, burns ~2 m per tick on ticks 2 and 3 (base speed burns 1 m)
    assert float(info["rubber_used"][0, 0]) == pytest.approx(4.0, abs=0.1)


def test_rubber_refills_over_rubber_time():
    env = make(rubber_m=5.0)
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 100, 30, LEFT)
    env.rubber[0, 0] = 0.0
    for _ in range(20):                           # 2 s in the open
        act(env, S_, S_)
    assert float(env.rubber[0, 0]) == pytest.approx(5.0 / 10.0 * 2.0, abs=0.02)


def test_head_on_is_fatal_even_with_rubber():
    env = make(rubber_m=5.0)
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 10, 12, LEFT)
    info = act(env, S_, S_)
    assert not env.alive[0].any()
    assert info["cause"][0].tolist() == [CAUSE_HEADON, CAUSE_HEADON]


# ---------------------------------------------------------------- walls over time
def test_wall_length_is_a_distance():
    env = make(wall_length_m=9.0)                 # 3 cells
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 100, 30, LEFT)
    for _ in range(5):                            # odometer 5: cells laid at 0..2 have expired
        act(env, S_, S_)
    assert [env.cell(0, 10, x) for x in range(10, 16)] == [EMPTY, EMPTY, EMPTY, 0, 0, 0]


def test_dead_cycle_walls_removed_after_delay():
    env = make(walls_stay_up_s=0.2)               # 6 ticks
    blank(env)
    env.place(0, 0, 0, 5, RIGHT)
    env.place(0, 1, 100, 30, LEFT)
    act(env, S_, S_)                              # moves to (0,6)
    act(env, L_, S_)                              # turns up into the rim, dies at tick 6
    assert not env.alive[0, 0]
    assert env.cell(0, 0, 5) == 0
    act(env, S_, S_)                              # tick 9: 3 ticks after death
    assert env.cell(0, 0, 5) == 0
    act(env, S_, S_)                              # tick 12: 6 ticks after death
    assert env.cell(0, 0, 5) == EMPTY and env.cell(0, 0, 6) == EMPTY


def test_living_cycle_head_never_expires():
    env = make(wall_length_m=3.0)
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 100, 30, LEFT)
    for _ in range(3):
        act(env, S_, S_)
    y, x = env.interior_pos(0, 0)
    assert env.cell(0, y, x) == 0


# ---------------------------------------------------------------- fortress rules
def _zone_cell(env, k):
    cy, cx = env.zone_center[k].tolist()
    return int(round(cy)) - env.P, int(round(cx)) - env.P


@pytest.mark.parametrize("att,dfn,seconds", [
    (1, 0, 5.0), (2, 0, 2.0), (2, 1, 10 / 3), (2, 2, 10.0), (1, 1, None), (0, 0, None), (0, 2, None),
])
def test_capture_times_match_the_wiki(att, dfn, seconds):
    """wiki.armagetronad.org Fortress: 1v0 5 s, 1v1 unconquerable, 2v0 2 s, 2v1 3.3 s, 2v2 10 s."""
    from tron.env import conquest_delta
    cfg = FortressConfig()
    per_step = float(conquest_delta(cfg, torch.tensor(float(att)), torch.tensor(float(dfn))))
    if seconds is None:
        assert per_step <= 0
    else:
        assert per_step > 0
        assert cfg.decision_s / per_step == pytest.approx(seconds, rel=1e-4)


def test_lone_attacker_in_the_simulator():
    env = make()
    blank(env)
    zy, zx = _zone_cell(env, 0)
    env.place(0, 1, zy, zx - 1, RIGHT)        # attacker in team 0's zone
    env.place(0, 0, 2, 2, RIGHT)              # defender far away
    act(env, S_, S_)
    assert env.progress[0, 0].item() == pytest.approx(0.02)   # 1/50 per decision: 5 s
    assert env.progress[0, 1].item() == 0.0


def test_one_defender_stops_one_attacker():
    env = make()
    blank(env)
    zy, zx = _zone_cell(env, 0)
    env.place(0, 1, zy, zx - 1, RIGHT)
    env.place(0, 0, zy + 2, zx - 2, RIGHT)
    env.progress[0, 0] = 0.5
    act(env, S_, S_)
    assert env.progress[0, 0].item() == 0.5     # 0.3 - 0.2 - 0.1 = 0: neither captured nor drained


def test_two_attackers_beat_one_defender():
    env = make(team_size=2)
    blank(env)
    zy, zx = _zone_cell(env, 0)
    env.place(0, 2, zy, zx - 2, RIGHT)
    env.place(0, 3, zy - 1, zx - 1, RIGHT)
    env.place(0, 0, zy + 1, zx, LEFT)
    env.place(0, 1, 5, 5, RIGHT)
    act(env, S_, S_, S_, S_)
    assert env.alive[0].all()
    assert env.progress[0, 0].item() == pytest.approx(0.03)   # (0.6 - 0.2 - 0.1) x 0.1 s


def test_empty_zone_drains():
    env = make()
    blank(env)
    env.place(0, 0, 10, 2, RIGHT)
    env.place(0, 1, 100, 30, LEFT)
    env.progress[0, 0] = 0.5
    act(env, S_, S_)
    assert env.progress[0, 0].item() == pytest.approx(0.49)


def test_conquest_wins_round():
    env = make()
    blank(env)
    zy, zx = _zone_cell(env, 0)
    env.place(0, 1, zy, zx - 1, RIGHT)
    env.place(0, 0, 2, 2, RIGHT)
    env.progress[0, 0] = 0.995
    info = act(env, S_, S_)
    assert info["done"][0]
    assert int(info["winner"][0]) == 1
    assert int(info["reason"][0]) == REASON_CONQUEST


def test_elimination_and_draw():
    env = make()
    blank(env)
    env.place(0, 0, 0, 5, UP)
    env.place(0, 1, 20, 20, LEFT)
    info = act(env, S_, S_)
    assert info["done"][0] and int(info["winner"][0]) == 1
    assert int(info["reason"][0]) == REASON_ELIMINATION

    env = make()
    blank(env)
    env.place(0, 0, 0, 5, UP)
    env.place(0, 1, 0, 20, UP)
    info = act(env, S_, S_)
    assert info["done"][0] and int(info["winner"][0]) == -1


def test_timeout_is_draw():
    env = make(max_time_s=0.3)
    for _ in range(2):
        info = act(env, S_, S_)
        assert not info["done"][0]
    info = act(env, S_, S_)
    assert info["done"][0] and int(info["winner"][0]) == -1
    assert int(info["reason"][0]) == REASON_TIMEOUT


def test_reset_restores_spawn():
    env = make(team_size=2, max_time_s=0.2)
    start = env.pos.clone()
    act(env, S_, S_, S_, S_)
    info = act(env, S_, S_, S_, S_)
    env.reset_done(info["done"])
    assert torch.equal(env.pos, start)
    assert env.alive.all()
    assert (env.owner >= 0).sum().item() == env.A
    assert env.steps[0].item() == 0 and env.progress.abs().sum().item() == 0


def test_envs_are_independent():
    env = make(n=2)
    blank(env, 0)
    blank(env, 1)
    for n in range(2):
        env.place(n, 0, 0, 5, RIGHT)
        env.place(n, 1, 30, 30, LEFT)
    a = torch.tensor([[L_, S_], [S_, S_]])
    env.step(a)
    assert not env.alive[0, 0] and env.alive[1, 0]
    assert env.interior_pos(1, 0) == (0, 6)


# ---------------------------------------------------------------- observations
def _crop(env, agent, n=0):
    codes, vec = env.observe()
    return codes[n, agent].long(), vec[n, agent]


def test_crop_rotation_facing_right():
    env = make()
    blank(env)
    R = env.cfg.obs_radius
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 25, 25, LEFT)
    env.set_cell(0, 10, 12, 1)     # 2 ahead
    env.set_cell(0, 13, 10, 1)     # 3 to the right (down)
    env.set_cell(0, 8, 10, 0)      # 2 to the left (up), own trail
    c, _ = _crop(env, 0)
    assert c[R - 2, R] // 3 == OCC_ENEMY
    assert c[R, R + 3] // 3 == OCC_ENEMY
    assert c[R, R - 2] // 3 == OCC_OWN
    assert c[R, R] // 3 == OCC_OWN


def test_crop_rotation_facing_up_and_left():
    env = make()
    blank(env)
    R = env.cfg.obs_radius
    env.place(0, 0, 15, 15, UP)
    env.place(0, 1, 15, 18, LEFT)  # enemy head 3 to the right of agent 0
    c, _ = _crop(env, 0)
    assert c[R, R + 3] // 3 == OCC_ENEMY_HEAD
    # agent 1 faces left; agent 0 is 3 cells ahead of it
    c1, _ = _crop(env, 1)
    assert c1[R - 3, R] // 3 == OCC_ENEMY_HEAD


def test_crop_sees_rim_and_zones():
    env = make()
    blank(env)
    R = env.cfg.obs_radius
    env.place(0, 0, 1, 10, UP)
    env.place(0, 1, 25, 25, LEFT)
    c, _ = _crop(env, 0)
    assert c[R - 2, R] // 3 == OCC_RIM
    assert c[R - 1, R] // 3 != OCC_RIM
    assert c[R, R] % 3 == ZONE_NONE


def test_teammate_codes():
    env = make(team_size=2)
    blank(env)
    R = env.cfg.obs_radius
    env.place(0, 0, 10, 10, UP)
    env.place(0, 1, 10, 12, UP)    # teammate head 2 right
    env.set_cell(0, 11, 12, 1)     # teammate trail 2 right, 1 behind
    env.place(0, 2, 30, 30, UP)
    c, _ = _crop(env, 0)
    assert c[R, R + 2] // 3 == 5   # OCC_TEAM_HEAD
    assert c[R + 1, R + 2] // 3 == OCC_TEAM


def test_vector_relative_positions_and_order():
    env = make(team_size=2)
    blank(env)
    S = env.S
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 10, 15, UP)    # teammate 5 ahead
    env.place(0, 2, 13, 10, LEFT)  # enemy 3 to the right
    env.place(0, 3, 30, 30, LEFT)
    _, v = _crop(env, 0)
    from tron.env import SELF_FEATURES, OTHER_FEATURES
    o = v[env.self_dim:].view(3, OTHER_FEATURES)
    # slot 0 = teammate (agent 1), slots 1,2 = enemies (agents 2,3)
    assert o[0, 2].item() == pytest.approx(5 / S) and o[0, 3].item() == pytest.approx(0.0)
    assert o[1, 2].item() == pytest.approx(0.0) and o[1, 3].item() == pytest.approx(3 / S)
    # relative heading: teammate faces UP while I face RIGHT -> (0-1)%4 = 3
    assert o[0, 4:8].tolist() == [0, 0, 0, 1]


def test_rays():
    env = make()
    blank(env)
    S = env.S
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 30, 30, LEFT)
    r = env.rays()[0, 0].tolist()
    assert r == [S - 1 - 10, 10, S - 1 - 10]


def test_vis_radius_hides_far_enemies_only():
    env = make(team_size=2, vis_radius=5)
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 30, 10, RIGHT)   # far teammate: still visible
    env.place(0, 2, 10, 13, RIGHT)   # near enemy: visible
    env.place(0, 3, 30, 30, RIGHT)   # far enemy: hidden
    from tron.env import SELF_FEATURES, OTHER_FEATURES
    _, v = _crop(env, 0)
    o = v[env.self_dim:].view(3, OTHER_FEATURES)
    assert o[:, 1].tolist() == [1.0, 1.0, 0.0]
    assert o[2, 2:].abs().sum().item() == 0.0
    assert o[2, 0].item() == 1.0     # alive status is still known


# ---------------------------------------------------------------- symmetry
@pytest.mark.parametrize("team_size", [1, 3, 7])
def test_mirror_symmetry_under_mirrored_play(team_size):
    """Team 1 is a 180-degree rotation of team 0. If both teams play the same
    actions, every observation must stay identical between agent i and agent T+i
    for the whole round. Catches rotation, masking and ordering bugs."""
    torch.manual_seed(0)
    env = make(team_size=team_size, n=8, explosion_radius_m=4.0, wall_length_m=400.0, walls_stay_up_s=8.0,
               rubber_m=5.0, reaction_delay=2, turn_cooldown=1)
    T = env.T
    for _ in range(80):
        codes, vec = env.observe()
        assert torch.equal(codes[:, :T], codes[:, T:])
        assert torch.allclose(vec[:, :T], vec[:, T:], atol=1e-5)
        a0 = torch.randint(0, 3, (env.N, T))
        info = env.step(torch.cat([a0, a0], 1))
        assert torch.equal(info["died"][:, :T], info["died"][:, T:])
        assert torch.equal(env.progress[:, 0], env.progress[:, 1])
        env.reset_done(info["done"])


# ---------------------------------------------------------------- explosions
def test_explosion_blows_hole_in_wall_that_was_hit():
    env = make(explosion_radius_m=6.0)
    blank(env)
    for y in range(4, 17):                 # enemy wall: column x=12, rows 4..16
        env.set_cell(0, y, 12, 1)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 30, 30, LEFT)
    act(env, S_, S_)                       # (10,11)
    info = act(env, S_, S_)                # crashes into (10,12): explosion centred there
    assert not env.alive[0, 0]
    col = [env.cell(0, y, 12) for y in range(4, 17)]
    # rows 8..12 are within 2 of (10,12); the rest of the wall survives
    assert col == [1, 1, 1, 1, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, 1, 1, 1, 1]
    # the dead cycle's own last cells are inside the blast too
    assert env.cell(0, 10, 11) == EMPTY and env.cell(0, 10, 10) == EMPTY
    assert int(info["blasted"][0, 0]) == 5


def test_explosion_disk_shape():
    env = make(explosion_radius_m=6.0)
    blank(env)
    for y in range(5, 16):
        for x in range(13, 24):
            env.set_cell(0, y, x, 1)
    env.place(0, 0, 10, 11, RIGHT)
    env.place(0, 1, 30, 30, LEFT)
    act(env, S_, S_)                       # moves to (10,12)
    info = act(env, S_, S_)                # crashes into (10,13)
    assert not env.alive[0, 0]
    for y in range(5, 16):
        for x in range(13, 24):
            inside = (y - 10) ** 2 + (x - 13) ** 2 <= 4
            assert (env.cell(0, y, x) == EMPTY) == inside, (y, x)


def test_explosion_never_breaks_rim():
    env = make(explosion_radius_m=6.0)
    blank(env)
    for x in range(2, 9):
        env.set_cell(0, 1, x, 1)           # enemy wall right next to the rim
    env.place(0, 0, 0, 5, UP)              # top row, heading into the rim
    env.place(0, 1, 30, 30, LEFT)
    act(env, S_, S_)
    assert not env.alive[0, 0]
    for x in range(0, 12):
        assert env.cell(0, -1, x) == RIM   # rim row untouched
    # crash point is (-1,5): wall cells at row 1 within 2 of it -> only (1,5)
    assert [env.cell(0, 1, x) for x in range(2, 9)] == [1, 1, 1, EMPTY, 1, 1, 1]
    assert env.cell(0, 0, 5) == EMPTY      # the dead cycle's own cell


def test_explosion_spares_living_cycle_but_not_its_trail():
    env = make(explosion_radius_m=6.0)
    blank(env)
    env.set_cell(0, 10, 12, RIM)           # something to crash into; blast centred on (10,12)
    env.place(0, 0, 10, 11, RIGHT)
    env.place(0, 1, 9, 12, UP)             # enemy right beside the blast, ends the step at (8,12)
    act(env, S_, S_)
    assert not env.alive[0, 0] and env.alive[0, 1]
    assert env.interior_pos(0, 1) == (8, 12)   # distance 2: inside the blast
    assert env.cell(0, 8, 12) == 1             # living head kept
    assert env.cell(0, 9, 12) == EMPTY         # its trail inside the blast is gone


def test_head_on_both_explode():
    env = make(explosion_radius_m=6.0)
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 10, 12, LEFT)
    env.set_cell(0, 8, 11, 0)
    env.set_cell(0, 13, 11, 1)             # 3 away from the centre: survives
    act(env, S_, S_)
    assert not env.alive[0].any()
    assert env.cell(0, 8, 11) == EMPTY
    assert env.cell(0, 10, 10) == EMPTY and env.cell(0, 10, 12) == EMPTY
    assert env.cell(0, 13, 11) == 1


def test_explosion_off_keeps_walls():
    env = make(explosion_radius_m=0)
    blank(env)
    env.set_cell(0, 10, 12, 1)
    env.set_cell(0, 9, 12, 1)
    env.place(0, 0, 10, 11, RIGHT)
    env.place(0, 1, 30, 30, LEFT)
    act(env, S_, S_)
    assert not env.alive[0, 0]
    assert env.cell(0, 10, 12) == 1 and env.cell(0, 9, 12) == 1 and env.cell(0, 10, 11) == 0


# ---------------------------------------------------------------- breach usage (measurement)
def _breach_setup(window=3.0):
    """Agent 0 crashes into an enemy wall (column x=12, owned by agent 2) and blows a hole
    at rows 8..12. Teammate agent 1 drives along row 9 toward the hole."""
    env = make(team_size=2, explosion_radius_m=6.0, breach_window_s=window)
    blank(env)
    for y in range(4, 17):
        env.set_cell(0, y, 12, 2)
    env.place(0, 0, 10, 11, RIGHT)         # crashes on the first step
    env.place(0, 1, 9, 7, RIGHT)
    env.place(0, 2, 30, 30, LEFT)
    env.place(0, 3, 30, 5, RIGHT)
    return env


def test_teammate_passing_through_hole_is_counted():
    env = _breach_setup()
    info = act(env, S_, S_, S_, S_)
    assert not env.alive[0, 0]
    assert bool(info["breach_made"][0, 0]) and not info["breach_used"][0, 0]
    for _ in range(4):                     # agent 1: x 9,10,11,12 -> enters the hole at x=12
        info = act(env, S_, S_, S_, S_)
    assert env.alive[0, 1] and env.interior_pos(0, 1) == (9, 12)
    assert bool(info["breach_used"][0, 0])
    assert int(info["breach_passes"][0, 0]) == 1
    info = act(env, S_, S_, S_, S_)        # x=13 is not part of the hole: no extra pass
    assert int(info["breach_passes"][0, 0]) == 1


def test_hole_used_too_late_is_not_counted():
    env = _breach_setup(window=4 / 30)
    for _ in range(5):
        info = act(env, S_, S_, S_, S_)
    assert env.interior_pos(0, 1) == (9, 12)
    assert bool(info["breach_made"][0, 0]) and not info["breach_used"][0, 0]
    assert int(info["breach_passes"][0, 0]) == 0


def test_enemy_passing_through_hole_counted_separately():
    env = make(team_size=2, explosion_radius_m=6.0)
    blank(env)
    for y in range(4, 17):
        env.set_cell(0, y, 12, 2)
    env.place(0, 0, 10, 11, RIGHT)
    env.place(0, 1, 30, 30, LEFT)
    env.place(0, 2, 9, 16, LEFT)           # enemy drives back through its own broken wall
    env.place(0, 3, 30, 5, RIGHT)
    for _ in range(4):
        info = act(env, S_, S_, S_, S_)
    assert env.interior_pos(0, 2) == (9, 12)
    assert not info["breach_used"][0, 0]
    assert int(info["breach_enemy_passes"][0, 0]) == 1


def test_no_breach_from_own_wall_or_rim():
    env = make(team_size=2, explosion_radius_m=6.0)
    blank(env)
    for y in range(4, 17):
        env.set_cell(0, y, 12, 1)          # teammate's wall
    env.place(0, 0, 10, 11, RIGHT)
    env.place(0, 1, 30, 30, LEFT)
    env.place(0, 2, 0, 5, UP)              # enemy hits the rim
    env.place(0, 3, 30, 5, RIGHT)
    info = act(env, S_, S_, S_, S_)
    assert not env.alive[0, 0] and not env.alive[0, 2]
    assert not info["breach_made"][0].any()


# ---------------------------------------------------------------- humanlike limits
def test_reaction_delay_returns_old_view():
    env = make(reaction_delay=2)
    blank(env)
    env.place(0, 0, 10, 10, RIGHT)
    env.place(0, 1, 30, 30, LEFT)
    views = []
    for _ in range(5):
        views.append(env.observe_now())
        delayed = env.observe()
        k = len(views) - 1
        expect = views[max(k - 2, 0)]
        a, b = env.pending_off, env.self_dim         # pending actions are current, the rest is delayed
        assert torch.equal(delayed[0], expect[0])
        assert torch.equal(delayed[1][..., :a], expect[1][..., :a])
        assert torch.equal(delayed[1][..., b:], expect[1][..., b:])
        act(env, S_, S_)


def test_reaction_delay_restarts_with_round():
    env = make(reaction_delay=2, max_time_s=0.3)
    for _ in range(3):
        env.observe()
        info = act(env, S_, S_)
    env.reset_done(info["done"])
    first = env.observe_now()
    d = env.observe()
    assert torch.equal(d[0], first[0])     # no view from the previous round leaks in


def test_observe_twice_same_step_is_idempotent():
    env = make(reaction_delay=1)
    for _ in range(3):
        a = env.observe()
        b = env.observe()
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        act(env, S_, S_)


def test_turn_cooldown_blocks_quick_second_turn():
    env = make(turn_cooldown=1)
    blank(env)
    env.place(0, 0, 10, 10, UP)
    env.place(0, 1, 30, 30, LEFT)
    act(env, R_, S_)                       # turn right: now facing RIGHT
    info = act(env, R_, S_)                # blocked: still RIGHT
    assert int(env.dir[0, 0]) == RIGHT and bool(info["turns_blocked"][0, 0])
    info = act(env, R_, S_)                # allowed again
    assert int(env.dir[0, 0]) == DOWN and not info["turns_blocked"][0, 0]


def test_no_cooldown_allows_instant_u_turn():
    env = make(turn_cooldown=0)
    blank(env)
    env.place(0, 0, 10, 10, UP)
    env.place(0, 1, 30, 30, LEFT)
    act(env, R_, S_)
    act(env, R_, S_)
    assert int(env.dir[0, 0]) == DOWN and env.alive[0, 0]


def test_invariants_hold_during_random_play():
    torch.manual_seed(1)
    env = make(team_size=3, n=16, explosion_radius_m=4.0, wall_length_m=60.0, walls_stay_up_s=0.5, rubber_m=5.0,
               reaction_delay=2, turn_cooldown=1)
    for _ in range(300):
        env.observe()
        info = env.step(torch.randint(0, 3, (env.N, env.A)))
        assert env.check_invariants() == []
        env.reset_done(info["done"])


# ---------------------------------------------------------------- agent id
def test_agent_slot_one_hot_in_observation():
    from tron.env import SELF_FEATURES
    env = make(team_size=3, agent_id_obs=True)
    _, vec = env.observe()
    ids = vec[0, :, SELF_FEATURES:SELF_FEATURES + 3]
    expect = torch.eye(3).repeat(2, 1)              # agent i and agent T+i share slot i
    assert torch.equal(ids, expect)
    assert env.vec_dim == SELF_FEATURES + 3 + 5 * 11


def test_agent_id_off_removes_it():
    from tron.env import SELF_FEATURES, OTHER_FEATURES
    env = make(team_size=3, agent_id_obs=False)
    _, vec = env.observe()
    assert vec.shape[-1] == SELF_FEATURES + 5 * OTHER_FEATURES == env.vec_dim


def test_old_checkpoint_configs_still_load():
    old = FortressConfig(team_size=2).to_dict()
    del old["agent_id_obs"]
    old["trail_ticks"] = 200                        # removed setting from an earlier version
    cfg = FortressConfig.from_dict(old)
    assert cfg.agent_id_obs is False and not hasattr(cfg, "trail_ticks")
    assert FortressConfig.from_dict(FortressConfig(team_size=2).to_dict()).agent_id_obs is True


# ---------------------------------------------------------------- map to scale
def test_map_scale_matches_the_real_game():
    """Measured in Retrocycles: end to end 16.55 s, zone to zone 10.35 s (at 30 m/s)."""
    cfg = FortressConfig()
    env = FortressEnv(cfg, 1)
    secs_per_cell = cfg.cell_m / cfg.cycle_speed
    assert env.S * secs_per_cell == pytest.approx(16.55, abs=0.3)
    centres = float(env.zone_center[0, 0] - env.zone_center[1, 0])
    assert (centres - 2 * cfg.zone_radius) * secs_per_cell == pytest.approx(10.35, abs=0.5)
    assert cfg.base_speed * cfg.ticks_per_step == 3000      # 1 cell per decision at 30 m/s


def test_v_formation_spawn():
    env = make(team_size=7)
    P = env.P
    pos = env.spawn_pos - P
    lead = pos[0]
    cy, cx = (env.zone_center[0] - P).tolist()
    assert int(lead[0]) == int(cy) and int(lead[1]) == int(cx) + 2     # 5 m beside the zone centre
    rel = (pos[:7] - lead).tolist()                                     # (dy, dx); +dy = behind team 0
    assert rel[1][0] >= 1 and rel[2][0] >= 1 and rel[1][1] == -rel[2][1] != 0   # first pair behind, either side
    for k in range(1, 6, 2):                                            # pairs mirror each other
        assert rel[k][0] == rel[k + 1][0] and rel[k][1] == -rel[k + 1][1]
    assert abs(rel[5][1]) > abs(rel[3][1]) > abs(rel[1][1])             # the V widens
    assert all(int(d) == 0 for d in env.spawn_dir[:7]) and all(int(d) == 2 for d in env.spawn_dir[7:])
    for k in range(7):                                                  # everyone starts in their own zone
        y, x = env.spawn_pos[k].tolist()
        assert int(env.zone_grid[y, x]) == 0


def test_bot_action_delay():
    env = make(reaction_delay=2)
    seen = []
    for k in range(5):
        a = torch.full((env.N, env.A), (k % 2) + 1, dtype=torch.long)    # 1,2,1,2,1
        seen.append(env.delay_actions(a)[0, 0].item())
        act(env, S_, S_)
    assert seen == [0, 0, 1, 0, 0]      # 2 steps late; the 2nd and 3rd turns were blocked while
                                        # the 1st was still pending (no stacking blind turns)


def test_bot_action_delay_restarts_with_round():
    env = make(reaction_delay=2, max_time_s=0.3)
    for k in range(3):
        env.delay_actions(torch.ones((env.N, env.A), dtype=torch.long))
        info = act(env, S_, S_)
    env.reset_done(info["done"])
    assert env.delay_actions(torch.full((env.N, env.A), 2, dtype=torch.long))[0, 0].item() == 0


def test_policy_sees_its_own_pending_actions():
    env = make(reaction_delay=2)
    blank(env)
    env.place(0, 0, 50, 50, UP)
    env.place(0, 1, 100, 30, LEFT)
    off = env.pending_off
    _, vec = env.observe()
    assert vec[0, 0, off:off + 6].tolist() == [1, 0, 0, 1, 0, 0]           # nothing pressed: straight x2
    act(env, L_, S_)
    _, vec = env.observe()
    assert vec[0, 0, off:off + 6].tolist() == [0, 1, 0, 1, 0, 0]           # most recent first: left
    act(env, R_, S_)
    _, vec = env.observe()
    assert vec[0, 0, off:off + 6].tolist() == [0, 0, 1, 0, 1, 0]           # right, then left


def test_bots_do_not_stack_blind_turns():
    env = make(reaction_delay=2)
    out = []
    for k in range(6):                                   # a bot that wants to turn left every step
        out.append(env.delay_actions(torch.full((env.N, env.A), L_, dtype=torch.long))[0, 0].item())
        act(env, S_, S_)
    # straight during the delay, then one left, then it waits until that turn has played out
    assert out == [0, 0, L_, 0, 0, L_]


# ---------------------------------------------------------------- other observation formats
@pytest.mark.parametrize("base_kw,view_kw", [
    (dict(reaction_delay=2, agent_id_obs=True), dict(reaction_delay=2, agent_id_obs=False)),
    (dict(reaction_delay=2), dict(reaction_delay=0, pending_actions_obs=False)),
    (dict(reaction_delay=0), dict(reaction_delay=2)),
    (dict(), dict(vis_radius=15.0)),
])
def test_obs_view_matches_an_env_built_with_that_format(base_kw, view_kw):
    """An ObsView of env A in format B must equal what an env built with format B sees,
    given identical play (observation settings never change the game itself)."""
    from tron.env import ObsView
    torch.manual_seed(0)
    a = make(team_size=3, n=4, rubber_m=5.0, explosion_radius_m=4.0, **base_kw)
    b = make(team_size=3, n=4, rubber_m=5.0, explosion_radius_m=4.0, **view_kw)
    view = ObsView(a, b.cfg)
    for _ in range(40):
        ca, va = view.observe()
        cb, vb = b.observe()
        assert torch.equal(ca, cb) and torch.allclose(va, vb)
        acts = torch.randint(0, 3, (4, a.A))
        ia, ib = a.step(acts), b.step(acts)
        assert torch.equal(ia["died"], ib["died"])
        a.reset_done(ia["done"]); b.reset_done(ib["done"])
