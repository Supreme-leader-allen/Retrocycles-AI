"""
Controllers and fixed-length match play, shared by training-time evaluation,
evaluate.py, run_baseline.py and render.py.

A controller is ``fn(env, codes, vec) -> (N, A) actions``; only the actions of
the team it is assigned to are used.
"""
import torch

from .bots import random_actions, heuristic_actions
from .metrics import MetricsTracker

SCRIPTED = ("random", "heuristic", "split")


def policy_controller(model, greedy=False):
    @torch.no_grad()
    def act(env, codes, vec):
        N, A = env.N, env.A
        logits, _ = model(codes.view(N * A, env.K, env.K), vec.view(N * A, -1))
        if greedy:
            a = logits.argmax(-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        return a.view(N, A)
    return act


def scripted_controller(name):
    if name == "random":
        return lambda env, codes, vec: random_actions(env)
    if name == "heuristic":
        return lambda env, codes, vec: heuristic_actions(env, split_roles=False)
    if name == "split":
        return lambda env, codes, vec: heuristic_actions(env, split_roles=True)
    raise ValueError(f"unknown controller {name!r}")


def play_rounds(env, ctrl_a, ctrl_b, rounds_per_env, names=("a", "b"), tracker=None,
                run_label="eval", csv_path=None, swap_half=True, on_step=None, samples=0):
    """
    Play exactly ``rounds_per_env`` rounds in every parallel env. Controller A
    plays team 0 and B team 1, except in the second half of the envs when
    ``swap_half`` (removes any side bias). Returns the list of metric rows,
    one per team per round (only the rows from each env's first
    ``rounds_per_env`` rounds are kept, so short rounds aren't over-counted).
    """
    N, T = env.N, env.T
    env.reset_done(torch.ones(N, dtype=torch.bool, device=env.device))
    if tracker is None:
        tracker = MetricsTracker(env, csv_path=None, run_label=run_label)
    else:
        for v in list(tracker.acc.values()) + list(tracker.agent_acc.values()):
            v.zero_()
    tracker.samples = samples
    a_is_team0 = torch.ones(N, dtype=torch.bool, device=env.device)
    if swap_half:
        a_is_team0[N // 2:] = False
    for n in range(N):
        t0 = names[0] if a_is_team0[n] else names[1]
        t1 = names[1] if a_is_team0[n] else names[0]
        tracker.controllers[n] = [t0, t1]
    team1 = (env.team == 1).unsqueeze(0)                                  # (1, A)
    a_mask = torch.where(a_is_team0.unsqueeze(1), ~team1, team1)          # agents controlled by A

    done_count = torch.zeros(N, dtype=torch.long)
    kept = []
    while (done_count < rounds_per_env).any():
        codes, vec = env.observe()
        act = torch.where(a_mask, ctrl_a(env, codes, vec), ctrl_b(env, codes, vec))
        info = env.step(act)
        if on_step is not None:
            on_step(env, info, a_mask)  # a_mask: (N, A) cycles driven by controller A
        tracker.update(info)
        if info["done"].any():
            rows = tracker.rows
            tracker.rows = []
            finished = info["done"].nonzero(as_tuple=True)[0].tolist()
            # tracker emits 2 rows per finished env, in env order
            for i, n in enumerate(finished):
                if done_count[n] < rounds_per_env:
                    kept.extend(rows[2 * i: 2 * i + 2])
                done_count[n] += 1
            env.reset_done(info["done"])
    tracker.rows = kept
    tracker.run_label = run_label
    for r in kept:
        r["run_label"] = run_label
    if csv_path:
        tracker.csv_path = csv_path
        import os, csv
        from .metrics import COLUMNS
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(COLUMNS)
    return tracker.flush()


def summarize(rows, controller):
    """Win/draw/loss rates for ``controller`` from a list of metric rows."""
    mine = [r for r in rows if r["controller"] == controller]
    n = max(len(mine), 1)
    w = sum(r["result"] == "win" for r in mine) / n
    d = sum(r["result"] == "draw" for r in mine) / n
    return {"rounds": len(mine), "win": w, "draw": d, "loss": 1 - w - d}
