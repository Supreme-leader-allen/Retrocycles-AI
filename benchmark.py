"""
Find the num_envs that gives the most training throughput on this machine.
Run this once on every new machine / pod type before a real run.

    python benchmark.py                    # 7v7, tries several num_envs
    python benchmark.py --envs 256 1024 4096

Prints samples/sec for the simulator alone and for full PPO iterations
(collect + update), plus peak GPU memory. Pick the largest num_envs before
throughput stops improving or memory runs out, then pass it to train.py with
--train num_envs=<N> (or NUM_ENVS=<N> for the experiment scripts).
"""
import argparse
import time

import torch

from tron.config import FortressConfig
from tron.env import FortressEnv
from tron.ppo import PPOTrainer, TrainConfig
from tron.rewards import RewardConfig


def sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-size", type=int, default=7)
    ap.add_argument("--envs", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    dev = args.device
    sizes = args.envs or ([256, 512, 1024, 2048, 4096] if dev.startswith("cuda") else [32, 64, 128])
    gcfg = FortressConfig(team_size=args.team_size)
    print(f"device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.startswith("cuda") else ""))
    print(f"{'num_envs':>9} {'sim sps':>12} {'train sps':>12} {'peak GB':>8}")
    for n in sizes:
        try:
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            env = FortressEnv(gcfg, n, device=dev)
            a = lambda: torch.randint(0, 3, (n, env.A), device=dev)
            for _ in range(3):
                env.observe(); info = env.step(a()); env.reset_done(info["done"])
            sync(dev); t = time.time()
            for _ in range(20):
                env.observe(); info = env.step(a()); env.reset_done(info["done"])
            sync(dev)
            sim = 20 * n * env.A / (time.time() - t)
            del env

            tcfg = TrainConfig(num_envs=n, eval_every=0)
            tr = PPOTrainer(gcfg, tcfg, RewardConfig(), dev)
            tr.train_iteration()  # warm-up
            sync(dev); t = time.time()
            for _ in range(args.iters):
                tr.train_iteration()
            sync(dev)
            train = args.iters * tcfg.rollout_len * n * tr.env.A / (time.time() - t)
            peak = torch.cuda.max_memory_allocated() / 1e9 if dev.startswith("cuda") else float("nan")
            print(f"{n:>9} {sim:>12,.0f} {train:>12,.0f} {peak:>8.2f}", flush=True)
            del tr
        except torch.cuda.OutOfMemoryError:
            print(f"{n:>9}  out of GPU memory, stop here")
            break


if __name__ == "__main__":
    main()
