"""
Record one round as an animated GIF so you can watch what the agents do.

    python render.py --a output/checkpoints/main_s0/latest.pt --b heuristic --gif replays/main_vs_heuristic.gif
    python render.py --a split --b heuristic --team-size 7 --gif replays/bots.gif

Colours: team 0 = blue (defends the bottom zone), team 1 = orange (top zone).
Bright squares are cycle heads, darker cells are trails, tinted circles are the
zones, and the bars at the top show each zone's conquest progress.
"""
import argparse

import torch

from tron.config import FortressConfig
from tron.env import FortressEnv
from tron.replay import record_round
from evaluate import make_controller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="team 0 (blue): random|heuristic|split|checkpoint")
    ap.add_argument("--b", required=True, help="team 1 (orange)")
    ap.add_argument("--gif", default="replays/round.gif")
    ap.add_argument("--team-size", type=int, default=7)
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ca, _, ck_a = make_controller(args.a, args.device, greedy=False)
    cb, _, ck_b = make_controller(args.b, args.device, greedy=False)
    ck = ck_a or ck_b
    gcfg = FortressConfig.from_dict(ck["game_cfg"]) if ck else FortressConfig(team_size=args.team_size)
    torch.manual_seed(args.seed)
    env = FortressEnv(gcfg, 1, device=args.device, seed=args.seed)
    frames, info = record_round(env, ca, cb, args.gif, scale=args.scale, fps=args.fps)
    w = int(info["winner"][0])
    reason = ["", "conquest", "elimination", "timeout"][int(info["reason"][0])]
    result = "draw" if w < 0 else f"team {w} ({'blue' if w == 0 else 'orange'}) wins"
    print(f"{len(frames) - 1} steps, {result} by {reason}\nsaved {args.gif}")


if __name__ == "__main__":
    main()
