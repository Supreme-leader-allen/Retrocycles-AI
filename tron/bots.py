"""
Scripted controllers used as baselines and evaluation opponents.

random      uniform over {straight, left, right}. The "no learning" control.
heuristic   avoids crashing (picks the direction with the most free space)
            and heads for the enemy zone. Every cycle does the same thing.
split       same, but cycles alternate roles: even slots attack the enemy
            zone, odd slots go home and defend. A hand-coded "coordinated"
            team: a reference point for what a simple division of labour
            looks like in the coordination metrics.
"""
import torch


def random_actions(env, gen=None):
    return torch.randint(0, 3, (env.N, env.A), device=env.device, generator=gen)


def heuristic_actions(env, split_roles=False, gen=None, noise=0.5):
    N, A = env.N, env.A
    rays = env.rays().float()                                              # (N, A, 3) straight/left/right
    team_of = env.team.unsqueeze(0).expand(N, A)
    target_team = 1 - team_of
    if split_roles:
        slot = (env.agent_ids % env.T).unsqueeze(0).expand(N, A)
        target_team = torch.where(slot % 2 == 1, team_of, target_team)
    tgt = env.zone_center[target_team]                                     # (N, A, 2)
    posf = env.pos.float()
    tf, tr = env._ego(tgt[..., 0] - posf[..., 0], tgt[..., 1] - posf[..., 1], env.dir)
    dist = torch.sqrt(tf * tf + tr * tr).clamp(min=1e-6)
    tf, tr = tf / dist, tr / dist
    align = torch.stack([tf, -tr, tr], dim=-1)                             # straight, left, right
    far = (dist > 0.6 * env.cfg.zone_radius).float().unsqueeze(-1)

    score = rays.clamp(max=10.0) + 3.0 * align * far * (rays >= 2).float()
    score[..., 0] += 0.3                                                    # mild preference for straight
    score = score - 100.0 * (rays == 0).float()
    if noise > 0:
        score = score + noise * torch.rand(score.shape, device=env.device, generator=gen)
    return score.argmax(-1)


def scripted(name):
    if name == "random":
        return lambda env: random_actions(env)
    if name == "heuristic":
        return lambda env: heuristic_actions(env, split_roles=False)
    if name == "split":
        return lambda env: heuristic_actions(env, split_roles=True)
    raise ValueError(f"unknown scripted controller {name!r}")
