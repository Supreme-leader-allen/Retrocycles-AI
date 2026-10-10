"""
Population-based training (PBT).

A population of members trains in generations. After each generation every
pair of members plays a round-robin tournament; fitness = average score vs the
other members (win 1, draw 0.5). The bottom ``bottom_frac`` of the population
is replaced by copies of random top members (EXPLOIT: weights, optimizer and
opponent pool are copied) with their hyperparameters multiplied by a random
factor in [0.8, 1.2] (EXPLORE).

Hyperparameters evolved: lr, ent_coef (PPO) and the conquest / kill shaping
weights (reward). Fitness is pure cross-play win rate, NOT a coordination
metric, so PBT can't "game" the thing being measured.

    python pbt.py --label pbt_s0 --members 4 --generations 10 --samples 2e9

``--samples`` is per member, so total compute = members x samples.
Re-running the same command resumes where it stopped.

Outputs (besides each member's normal training outputs, labelled <label>_m<k>):
    checkpoints/<label>/members/m<k>/latest.pt   each member
    checkpoints/<label>/latest.pt                copy of the best member after the last generation
    checkpoints/<label>/pbt_state.json           resume state
    logs/<label>_generations.jsonl               one line per generation: hparams, fitness, lineage
    metrics/<label>_crossplay.csv                every cross-play and vs-heuristic round
"""
import argparse
import json
import math
import os
import random
import shutil

import torch

from tron.cli import parse_overrides, pick_device
from tron.config import FortressConfig
from tron.env import FortressEnv
from tron.ppo import PPOTrainer, TrainConfig, load_policy
from tron.rewards import RewardConfig
from tron.runner import TrainingRun, write_run_info
from tron.play import play_rounds, policy_controller, scripted_controller, summarize

# name: (lower bound, upper bound)
HPARAM_BOUNDS = {
    "lr": (1e-5, 2e-3),
    "ent_coef": (1e-4, 0.1),
    "conquest": (0.0, 5.0),
    "kill": (0.0, 1.0),
}


def clip_hp(hp):
    return {k: min(max(v, HPARAM_BOUNDS[k][0]), HPARAM_BOUNDS[k][1]) for k, v in hp.items()}


def initial_hparams(base, k, rng):
    """Member 0 keeps the defaults; the others start spread out by up to 2x in either direction."""
    if k == 0:
        return dict(base)
    return clip_hp({n: v * math.exp(rng.uniform(-math.log(2), math.log(2))) for n, v in base.items()})


def perturb(hp, rng, lo, hi):
    return clip_hp({n: v * rng.uniform(lo, hi) for n, v in hp.items()})


def member_dir(out, label, k):
    return os.path.join(out, "checkpoints", label, "members", f"m{k}")


def train_member(args, k, hp, gcfg, tcfg_base, rcfg_base, target, device):
    tcfg = TrainConfig(**tcfg_base.to_dict())
    rcfg = RewardConfig(**rcfg_base.to_dict())
    tcfg.seed = tcfg_base.seed * 100 + k
    tcfg.total_samples = target
    mlabel = f"{args.label}_m{k}"
    trainer = PPOTrainer(gcfg, tcfg, rcfg, device,
                         metrics_csv=os.path.join(args.out, "metrics", f"{mlabel}.csv"), run_label=mlabel)
    ckpt_path = os.path.join(member_dir(args.out, args.label, k), "latest.pt")
    if os.path.exists(ckpt_path):
        trainer.load(torch.load(ckpt_path, map_location=device, weights_only=False))
    trainer.set_hparams(**hp)
    if trainer.samples >= target:
        return
    run = TrainingRun(trainer, mlabel, args.out, ckpt_dir=member_dir(args.out, args.label, k))
    if not os.path.exists(os.path.join(run.ckpt_dir, "run_info.json")):
        write_run_info(os.path.join(run.ckpt_dir, "run_info.json"), trainer, pbt_label=args.label, member=k)
    try:
        run.run_until(target)
    finally:
        run.close()
    del trainer, run
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def crossplay(args, gcfg, members, gen, samples, device):
    """Round-robin between members (+ each member vs the heuristic bot, logged but not fitness)."""
    env = FortressEnv(gcfg, args.eval_envs, device=device, seed=1000 + gen)
    csv_path = os.path.join(args.out, "metrics", f"{args.label}_crossplay.csv")
    models = {k: load_policy(os.path.join(member_dir(args.out, args.label, k), "latest.pt"), device)[0]
              for k in members}
    score = {k: [] for k in members}
    vs_heur = {}
    per_env = max(1, -(-args.eval_rounds // args.eval_envs))
    for i in members:
        for j in members:
            if j <= i:
                continue
            rows = play_rounds(env, policy_controller(models[i]), policy_controller(models[j]), per_env,
                               names=(f"m{i}", f"m{j}"), run_label=f"{args.label}_gen{gen}",
                               csv_path=csv_path, samples=samples)
            si, sj = summarize(rows, f"m{i}"), summarize(rows, f"m{j}")
            score[i].append(si["win"] + 0.5 * si["draw"])
            score[j].append(sj["win"] + 0.5 * sj["draw"])
        rows = play_rounds(env, policy_controller(models[i]), scripted_controller("heuristic"), per_env,
                           names=(f"m{i}", "heuristic"), run_label=f"{args.label}_gen{gen}",
                           csv_path=csv_path, samples=samples)
        vs_heur[i] = summarize(rows, f"m{i}")["win"]
    fitness = {k: (sum(v) / len(v) if v else 0.0) for k, v in score.items()}
    return fitness, vs_heur


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", "output"))
    ap.add_argument("--team-size", type=int, default=7)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--samples", type=float, required=True, help="agent-steps PER MEMBER over all generations")
    ap.add_argument("--eval-rounds", type=int, default=128, help="rounds per cross-play pairing")
    ap.add_argument("--eval-envs", type=int, default=64)
    ap.add_argument("--bottom-frac", type=float, default=0.25)
    ap.add_argument("--perturb", type=float, nargs=2, default=(0.8, 1.2))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--game", nargs="*", action="extend")
    ap.add_argument("--train", nargs="*", action="extend")
    ap.add_argument("--reward", nargs="*", action="extend")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    rng = random.Random(args.seed)
    root = os.path.join(args.out, "checkpoints", args.label)
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    state_path = os.path.join(root, "pbt_state.json")
    gen_log = os.path.join(args.out, "logs", f"{args.label}_generations.jsonl")

    if os.path.exists(state_path):
        with open(state_path) as f:
            st = json.load(f)
        gcfg = FortressConfig.from_dict(st["game"])
        tcfg = TrainConfig(**st["train"])
        rcfg = RewardConfig(**st["reward"])
        hparams = {int(k): v for k, v in st["hparams"].items()}
        start_gen = st["generation"]
        rng.setstate(tuple(st["rng"][:1]) + (tuple(st["rng"][1]),) + tuple(st["rng"][2:]))
        print(f"resuming PBT {args.label} at generation {start_gen}")
    else:
        gcfg = FortressConfig(team_size=args.team_size, **parse_overrides(args.game, FortressConfig))
        tcfg = TrainConfig(**parse_overrides(args.train, TrainConfig))
        tcfg.seed = args.seed
        rcfg = RewardConfig(**parse_overrides(args.reward, RewardConfig))
        base = {"lr": tcfg.lr, "ent_coef": tcfg.ent_coef, "conquest": rcfg.conquest, "kill": rcfg.kill}
        hparams = {k: initial_hparams(base, k, rng) for k in range(args.members)}
        start_gen = 0

    members = sorted(hparams)
    gen_samples = args.samples / args.generations
    print(f"PBT {args.label}: {len(members)} members x {args.generations} generations x "
          f"{gen_samples:,.0f} samples = {len(members) * args.samples:,.0f} total, device {device}")

    for gen in range(start_gen, args.generations):
        target = int(round((gen + 1) * gen_samples))
        for k in members:
            print(f"--- generation {gen} member m{k} {hparams[k]}", flush=True)
            train_member(args, k, hparams[k], gcfg, tcfg, rcfg, target, device)

        fitness, vs_heur = crossplay(args, gcfg, members, gen, target, device)
        ranked = sorted(members, key=lambda k: fitness[k], reverse=True)
        record = {"generation": gen, "samples_per_member": target, "best": ranked[0],
                  "members": [{"id": k, "hparams": hparams[k], "fitness": fitness[k],
                               "win_vs_heuristic": vs_heur[k]} for k in members],
                  "replaced": []}
        print("fitness: " + "  ".join(f"m{k} {fitness[k]:.3f} (heur {vs_heur[k]:.2f})" for k in ranked), flush=True)

        best_src = os.path.join(member_dir(args.out, args.label, ranked[0]), "latest.pt")
        shutil.copyfile(best_src, os.path.join(root, "latest.pt"))

        if gen < args.generations - 1 and len(members) > 1:
            n_b = max(1, int(len(members) * args.bottom_frac))
            top, bottom = ranked[:n_b], ranked[-n_b:]
            for b in bottom:
                src = rng.choice(top)
                shutil.copyfile(os.path.join(member_dir(args.out, args.label, src), "latest.pt"),
                                os.path.join(member_dir(args.out, args.label, b), "latest.pt"))
                old = hparams[b]
                hparams[b] = perturb(hparams[src], rng, *args.perturb)
                record["replaced"].append({"member": b, "copied_from": src, "old_hparams": old,
                                           "new_hparams": hparams[b]})
                print(f"    m{b} <- m{src}, new hparams {hparams[b]}", flush=True)

        with open(gen_log, "a") as f:
            f.write(json.dumps(record) + "\n")
        st = {"generation": gen + 1, "hparams": hparams, "game": gcfg.to_dict(), "train": tcfg.to_dict(),
              "reward": rcfg.to_dict(), "rng": list(rng.getstate()[:1]) + [list(rng.getstate()[1])] +
              list(rng.getstate()[2:])}
        tmp = state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, state_path)

    print(f"PBT done; best member copied to {os.path.join(root, 'latest.pt')}")


if __name__ == "__main__":
    main()
