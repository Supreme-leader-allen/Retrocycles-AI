"""
Scripted controllers used as baselines, evaluation opponents and training opponents.

random      uniform over {straight, left, right}. The "no learning" control.
heuristic   all-out attack: every cycle heads for the enemy zone, but the team fans
            out into flank lanes first (from the V spawn they would otherwise drive
            up the middle in one block). Avoids crashing by picking the direction
            with the most free space.
split       the positions from the Armagetron wiki's Fortress page: slot 1 center,
            2-3 attackers (via the flanks), 4-5 sweepers (support in your own half),
            6-7 defenders (stay in your own zone). A hand-coded "coordinated" team.

Both scripted bots read the game state directly; play.py and the trainer delay their actions
by the same reaction_delay as the policy's observations (env.delay_actions).
"""
import torch

ROLE_ATTACK, ROLE_CENTER, ROLE_SWEEP, ROLE_DEFEND = range(4)
WIKI_ROLES = [ROLE_CENTER, ROLE_ATTACK, ROLE_ATTACK, ROLE_SWEEP, ROLE_SWEEP, ROLE_DEFEND, ROLE_DEFEND]

LANE_SPACING = 15        # cells between flank lanes (45 m)
LANE_UNTIL = 30          # cells from the enemy zone centre at which flankers turn in
LANE_LOOKAHEAD = 10      # flankers aim this many cells ahead, in their lane
SWEEP_DEPTH = 0.3        # sweepers hold this fraction of the way from own zone to enemy zone
SWEEP_WIDTH = 20         # cells to the side for sweepers


def random_actions(env, gen=None):
    return torch.randint(0, 3, (env.N, env.A), device=env.device, generator=gen)


def _targets(env, roles):
    """Target cell (padded coords) for every agent, (N, A, 2) float, given a role per slot (T,)."""
    N, A, T = env.N, env.A, env.T
    team = env.team                                                  # (A,)
    slot = env.agent_ids % T
    own = env.zone_center[team]                                      # (A, 2)
    foe = env.zone_center[1 - team]
    toward = torch.sign(foe[:, 0] - own[:, 0])                       # +1 / -1 along y
    # side of the slot in the V: odd slots one way, even slots the other, further out each pair;
    # mirrored for team 1 so the layout is point-symmetric like the map
    away = ((slot + 1) // 2).float()
    sign = torch.where(slot % 2 == 1, 1.0, -1.0) * torch.where(slot == 0, 0.0, 1.0)
    sign = sign * torch.where(team == 0, 1.0, -1.0)
    role = roles[slot]                                               # (A,)

    pos = env.pos.float()                                            # (N, A, 2)
    tgt = foe.unsqueeze(0).expand(N, A, 2).clone()                   # default: enemy zone centre
    # attackers: drive up a flank lane (aiming a few cells ahead in the lane, so they actually
    # steer over to it) until close to the enemy zone, then go for its centre
    lane = torch.stack([pos[..., 0] + toward * LANE_LOOKAHEAD,
                        (foe[:, 1] + sign * away * LANE_SPACING).expand(N, A)], dim=-1)
    far = (pos[..., 0] - foe[:, 0]).abs() > LANE_UNTIL               # (N, A)
    attack = (role == ROLE_ATTACK).unsqueeze(0)
    tgt = torch.where((attack & far).unsqueeze(-1), lane, tgt)
    # sweepers: hold a point in their own half, to the side
    sweep_pt = own.clone()
    sweep_pt[:, 0] = own[:, 0] + toward * SWEEP_DEPTH * (foe[:, 0] - own[:, 0]).abs()
    sweep_pt[:, 1] = own[:, 1] + sign * SWEEP_WIDTH
    sweep = (role == ROLE_SWEEP).view(1, A, 1)
    tgt = torch.where(sweep, sweep_pt.unsqueeze(0).expand(N, A, 2), tgt)
    # defenders: own zone centre
    defend = (role == ROLE_DEFEND).view(1, A, 1)
    tgt = torch.where(defend, own.unsqueeze(0).expand(N, A, 2), tgt)
    return tgt


def heuristic_actions(env, split_roles=False, gen=None, noise=0.5):
    """split_roles: bool for every env, or an (N,) bool tensor choosing per env
    (False = all-attack heuristic, True = wiki positions)."""
    N, A, T = env.N, env.A, env.T
    dev = env.device
    attack_roles = torch.full((T,), ROLE_ATTACK, dtype=torch.long, device=dev)
    wiki = torch.tensor([WIKI_ROLES[k % len(WIKI_ROLES)] for k in range(T)], dtype=torch.long, device=dev)
    if torch.is_tensor(split_roles):
        tgt = torch.where(split_roles.view(N, 1, 1), _targets(env, wiki), _targets(env, attack_roles))
    else:
        tgt = _targets(env, wiki if split_roles else attack_roles)

    rays = env.rays().float()                                        # (N, A, 3) straight/left/right
    posf = env.pos.float()
    tf, tr = env._ego(tgt[..., 0] - posf[..., 0], tgt[..., 1] - posf[..., 1], env.dir)
    dist = torch.sqrt(tf * tf + tr * tr).clamp(min=1e-6)
    tf, tr = tf / dist, tr / dist
    align = torch.stack([tf, -tr, tr], dim=-1)                       # straight, left, right
    far = (dist > 0.6 * env.cfg.zone_radius).float().unsqueeze(-1)

    # plan ahead for the reaction delay: a move decided now happens reaction_delay decisions
    # later, so any direction with less free space than (delay + 1) decisions of travel at the
    # current speed is dangerous
    from .config import SPEED_UNIT
    cells_per_step = env.speed.float() / SPEED_UNIT * env.cfg.ticks_per_step          # (N, A)
    margin = ((env.cfg.reaction_delay + 1) * cells_per_step).unsqueeze(-1)
    score = rays.clamp(max=10.0) + 3.0 * align * far * (rays >= margin + 1).float()
    score[..., 0] += 0.3                                              # mild preference for straight
    score = score - 100.0 * (rays == 0).float() - 50.0 * (rays < margin).float()
    if noise > 0:
        score = score + noise * torch.rand(score.shape, device=dev, generator=gen)
    return score.argmax(-1)


def scripted(name):
    if name == "random":
        return lambda env: random_actions(env)
    if name == "heuristic":
        return lambda env: heuristic_actions(env, split_roles=False)
    if name == "split":
        return lambda env: heuristic_actions(env, split_roles=True)
    raise ValueError(f"unknown scripted controller {name!r}")
