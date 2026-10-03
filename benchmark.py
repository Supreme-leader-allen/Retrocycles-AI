"""
Find the num_envs that gives the most training throughput on this machine.
Run this once on every new machine / pod type before a real run.

    python benchmark.py                    # 7v7, tries several num_envs
    python benchmark.py --envs 256 1024 4096

Prints samples/sec for the simulator alone and for full PPO iterations
(collect + update), how the time splits between collecting games and
updating the network, and peak GPU memory. On CUDA it also re-runs the best
size without mixed precision so you can see what bfloat16 buys.

Pick the num_envs with the best "train sps" (ties: the smaller one, it logs
and evaluates more often), then: export NUM_ENVS=<that number>
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


def bench_train(gcfg, n, dev, iters, amp=True):
    tcfg = TrainConfig(num_envs=n, eval_every=0, replay_every=0, check_every=0, amp=amp)
    tr = PPOTrainer(gcfg, tcfg, RewardConfig(), dev)
    tr.train_iteration()  # warm-up (cudnn autotuning)
    sync(dev)
    t = time.time()
    collect = update = 0.0
    for _ in range(iters):
        s = tr.train_iteration()
        collect += s["collect_s"]
        update += s["update_s"]
    sync(dev)
    sps = iters * tcfg.rollout_len * n * tr.env.A / (time.time() - t)
    del tr
    return sps, collect / max(collect + update, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-size", type=int, default=7)
    ap.add_argument("--envs", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    dev = args.device
    cuda = dev.startswith("cuda")
    sizes = args.envs or ([256, 512, 1024, 2048, 4096] if cuda else [32, 64, 128])
    gcfg = FortressConfig(team_size=args.team_size)
    print(f"device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""))
    print(f"{'num_envs':>9} {'sim sps':>12} {'train sps':>12} {'collect %':>10} {'peak GB':>8}")
    best = (0.0, None)
    for n in sizes:
        try:
            if cuda:
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
            train, cfrac = bench_train(gcfg, n, dev, args.iters)
            peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else float("nan")
            print(f"{n:>9} {sim:>12,.0f} {train:>12,.0f} {100 * cfrac:>9.0f}% {peak:>8.2f}", flush=True)
            if train > best[0] * 1.05:   # only prefer a bigger size if it is clearly faster
                best = (train, n)
        except torch.cuda.OutOfMemoryError:
            print(f"{n:>9}  out of GPU memory, stop here")
            break
    if cuda and best[1]:
        torch.cuda.empty_cache()
        off, _ = bench_train(gcfg, best[1], dev, args.iters, amp=False)
        print(f"\nmixed precision OFF at num_envs={best[1]}: {off:,.0f} train sps "
              f"(ON is {best[0] / max(off, 1):.1f}x faster)")
    if best[1]:
        print(f"\nsuggested: export NUM_ENVS={best[1]}")
        print(f"hours per 1e9 samples at {best[0]:,.0f} sps: {1e9 / best[0] / 3600:.1f}")


if __name__ == "__main__":
    main()
