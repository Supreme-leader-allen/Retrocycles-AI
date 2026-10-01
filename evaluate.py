"""
Play a fixed number of rounds between two controllers and write per-round
coordination metrics to CSV. Sides are swapped for half of the rounds.

Controllers: random | heuristic | split | path/to/checkpoint.pt

    # trained policy vs the heuristic bot, 1024 rounds
    python evaluate.py --a output/checkpoints/main/latest.pt --b heuristic --label main_vs_heuristic

    # the no-learning control (this is the "baseline_random" equivalent)
    python evaluate.py --a random --b random --label baseline_random --team-size 7

    # head-to-head between two ablations
    python evaluate.py --a output/checkpoints/main/latest.pt --b output/checkpoints/no_credit/latest.pt --label main_vs_no_credit
"""
import argparse
import os

import torch

from tron.config import FortressConfig
from tron.env import FortressEnv
from tron.ppo import load_policy
from tron.play import play_rounds, policy_controller, scripted_controller, summarize, SCRIPTED


def make_controller(spec, device, greedy):
    if spec in SCRIPTED:
        return scripted_controller(spec), spec, None
    model, ckpt = load_policy(spec, device)
    name = os.path.basename(os.path.dirname(os.path.abspath(spec))) or spec
    return policy_controller(model, greedy=greedy), name, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", "output"))
    ap.add_argument("--rounds", type=int, default=1024, help="total rounds (spread over --envs)")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--team-size", type=int, default=None, help="needed only if neither side is a checkpoint")
    ap.add_argument("--vis-radius", type=float, default=None, help="override the game's vis_radius")
    ap.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    ca, name_a, ck_a = make_controller(args.a, args.device, args.greedy)
    cb, name_b, ck_b = make_controller(args.b, args.device, args.greedy)
    if name_a == name_b:
        name_a, name_b = name_a + "_A", name_b + "_B"
    ck = ck_a or ck_b
    if ck is not None:
        gcfg = FortressConfig(**ck["game_cfg"])
        if ck_a and ck_b and ck_a["game_cfg"] != ck_b["game_cfg"]:
            print("note: the two checkpoints were trained with different game configs; using A's")
    else:
        gcfg = FortressConfig(team_size=args.team_size or 7)
    if args.vis_radius is not None:
        gcfg.vis_radius = args.vis_radius

    torch.manual_seed(args.seed)
    n_envs = max(2, min(args.envs, args.rounds))
    env = FortressEnv(gcfg, n_envs, device=args.device, seed=args.seed)
    per_env = max(1, -(-args.rounds // n_envs))
    csv_path = os.path.join(args.out, "metrics", f"{args.label}.csv")
    rows = play_rounds(env, ca, cb, per_env, names=(name_a, name_b), run_label=args.label, csv_path=csv_path)
    for name in (name_a, name_b):
        s = summarize(rows, name)
        print(f"{name:>30}: {s['rounds']} rounds  win {s['win']:.3f}  draw {s['draw']:.3f}  loss {s['loss']:.3f}")
    print(f"wrote {len(rows)} rows -> {csv_path}")


if __name__ == "__main__":
    main()
