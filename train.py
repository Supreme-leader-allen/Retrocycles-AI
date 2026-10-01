"""
Train a Fortress team with PPO self-play.

    python train.py --label main --team-size 7 --samples 2e9
    python train.py --label main --resume            # continue from OUT/checkpoints/main/latest.pt

Any config field can be overridden with --game / --train / --reward key=value, e.g.
    python train.py --label test --team-size 2 --train num_envs=64 rollout_len=32 --samples 2e6

Everything a run writes is listed in tron/runner.py and DATA.md.
"""
import argparse
import os
import sys

import torch

from tron.cli import parse_overrides, pick_device
from tron.config import FortressConfig
from tron.ppo import PPOTrainer, TrainConfig
from tron.rewards import RewardConfig
from tron.runner import TrainingRun, write_run_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="run name; also the run_label in metric CSVs")
    ap.add_argument("--out", default=os.environ.get("OUT", "output"))
    ap.add_argument("--team-size", type=int, default=7)
    ap.add_argument("--samples", type=float, default=None, help="total agent-steps to train for")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None, help="start weights from this checkpoint (fresh optimizer/counters)")
    ap.add_argument("--game", nargs="*", help="FortressConfig overrides key=value")
    ap.add_argument("--train", nargs="*", help="TrainConfig overrides key=value")
    ap.add_argument("--reward", nargs="*", help="RewardConfig overrides key=value")
    args = ap.parse_args()

    device = pick_device(args.device)
    latest = os.path.join(args.out, "checkpoints", args.label, "latest.pt")

    ckpt = None
    if args.resume:
        if not os.path.exists(latest):
            sys.exit(f"--resume given but {latest} does not exist")
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        gcfg = FortressConfig(**ckpt["game_cfg"])
        tcfg = TrainConfig(**ckpt["train_cfg"])
        rcfg = RewardConfig(**ckpt["reward_cfg"])
        for k, v in parse_overrides(args.train, TrainConfig).items():  # e.g. a different num_envs on a new pod
            setattr(tcfg, k, v)
    else:
        gcfg = FortressConfig(team_size=args.team_size, **parse_overrides(args.game, FortressConfig))
        tcfg = TrainConfig(**parse_overrides(args.train, TrainConfig))
        rcfg = RewardConfig(**parse_overrides(args.reward, RewardConfig))
    if args.samples is not None:
        tcfg.total_samples = args.samples

    metrics_csv = os.path.join(args.out, "metrics", f"{args.label}.csv")
    trainer = PPOTrainer(gcfg, tcfg, rcfg, device, metrics_csv=metrics_csv, run_label=args.label)
    if ckpt is not None:
        trainer.load(ckpt)
        print(f"resumed {args.label} at iteration {trainer.iteration}, {trainer.samples:,} samples")
    elif args.init_from:
        src = torch.load(args.init_from, map_location=device, weights_only=False)
        trainer.model.load_state_dict(src["model"])
        print(f"initialised weights from {args.init_from}")

    run = TrainingRun(trainer, args.label, args.out)
    write_run_info(os.path.join(run.ckpt_dir, "run_info.json"), trainer,
                   resumed_at=trainer.samples if ckpt is not None else None)

    env = trainer.env
    n_params = sum(p.numel() for p in trainer.model.parameters())
    per_iter = tcfg.rollout_len * tcfg.num_envs * env.A
    print(f"[{args.label}] {gcfg.team_size}v{gcfg.team_size}, arena {gcfg.arena_size}, device {device}, "
          f"{tcfg.num_envs} envs, {per_iter:,} samples/iter, {n_params:,} params, "
          f"target {tcfg.total_samples:,.0f} samples", flush=True)
    try:
        run.run_until(tcfg.total_samples)
    except KeyboardInterrupt:
        pass
    finally:
        run.close()
    print(f"done: {trainer.samples:,} samples -> {latest}")


if __name__ == "__main__":
    main()
