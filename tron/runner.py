"""
The training loop shared by train.py and pbt.py: iterate PPO, and around it
record everything needed for debugging and for the paper.

Per iteration  -> logs/<label>_train.csv            (losses, speed, action mix, reward parts, deaths...)
Per round      -> metrics/<label>.csv               (coordination metrics of every training round)
Every eval     -> metrics/<label>_eval.csv          (rounds vs heuristic and vs split bots)
               -> heatmaps/<label>/vs_heuristic_<samples>.npz   (where cycles drive / die)
Every replay   -> replays/<label>/<samples>_vs_heuristic.gif
Every save     -> checkpoints/<label>/latest.pt, ckpt_<samples>.pt
On a crash     -> checkpoints/<label>/debug_<reason>_<iteration>.pt (model + full game state)
Once           -> checkpoints/<label>/run_info.json (configs, versions, git commit, GPU)
"""
import csv
import json
import os
import platform
import subprocess
import sys
import time

import torch

from .env import FortressEnv
from .play import play_rounds, policy_controller, scripted_controller, summarize
from .replay import HeatmapRecorder, record_round

EVAL_OPPONENTS = ("heuristic", "split")


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def git_commit():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def write_run_info(path, trainer, **extra):
    info = {
        "game": trainer.gcfg.to_dict(), "train": trainer.cfg.to_dict(), "reward": trainer.rcfg.to_dict(),
        "device": str(trainer.device),
        "gpu": torch.cuda.get_device_name(0) if trainer.device.type == "cuda" else None,
        "torch": torch.__version__, "python": sys.version.split()[0], "platform": platform.platform(),
        "git_commit": git_commit(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": sum(p.numel() for p in trainer.model.parameters()),
        "argv": sys.argv,
    }
    info.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)


class TrainingRun:
    def __init__(self, trainer, label, out, verbose=True, ckpt_dir=None):
        self.tr = trainer
        self.label = label
        self.out = out
        self.verbose = verbose
        self.ckpt_dir = ckpt_dir or os.path.join(out, "checkpoints", label)
        self.latest = os.path.join(self.ckpt_dir, "latest.pt")
        self.log_csv = os.path.join(out, "logs", f"{label}_train.csv")
        self.eval_csv = os.path.join(out, "metrics", f"{label}_eval.csv")
        for d in (self.ckpt_dir, os.path.join(out, "logs"), os.path.join(out, "metrics")):
            os.makedirs(d, exist_ok=True)
        tcfg, dev = trainer.cfg, trainer.device
        self.eval_env = (FortressEnv(trainer.gcfg, tcfg.eval_envs, device=dev, seed=tcfg.seed + 1)
                         if tcfg.eval_every > 0 else None)
        self.replay_env = (FortressEnv(trainer.gcfg, 1, device=dev, seed=tcfg.seed + 2)
                           if tcfg.replay_every > 0 else None)
        self.eval_keys = [f"eval_{o}_{k}" for o in EVAL_OPPONENTS for k in ("win", "draw", "role_spec", "breach_use")]
        self._log_file = None
        self._writer = None

    # ------------------------------------------------------------------ saving
    def save(self, tag=None):
        state = self.tr.state()
        tmp = self.latest + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.latest)
        if tag is not None:
            # history checkpoints are for evaluation, not resuming: skip the opponent pool (~30 copies
            # of the network) so each file is ~10x smaller. latest.pt keeps it.
            slim = {k: v for k, v in state.items() if k not in ("pool", "opt")}
            torch.save(slim, os.path.join(self.ckpt_dir, f"ckpt_{tag}.pt"))

    def save_debug(self, reason):
        """Everything needed to reproduce a crash: model, optimizer and the full game state."""
        env = self.tr.env
        state = self.tr.state()
        state["env_state"] = {k: getattr(env, k).detach().cpu() for k in (
            "owner", "stamp", "pos", "dir", "speed", "credit", "alive", "death_tick", "death_cause",
            "progress", "tick", "steps", "breach_by")}
        path = os.path.join(self.ckpt_dir, f"debug_{reason}_{self.tr.iteration}.pt")
        torch.save(state, path)
        print(f"!!! saved debug state to {path}", flush=True)

    # ------------------------------------------------------------------ evaluation
    def evaluate(self):
        tr, tcfg = self.tr, self.tr.cfg
        tr.model.eval()
        stats = {}
        for opp in EVAL_OPPONENTS:
            hm = HeatmapRecorder(self.eval_env)
            rows = play_rounds(self.eval_env, policy_controller(tr.model), scripted_controller(opp),
                               tcfg.eval_rounds, names=("policy", opp), run_label=f"{self.label}_vs_{opp}",
                               csv_path=self.eval_csv, samples=tr.samples, on_step=hm)
            s = summarize(rows, "policy")
            mine = [r for r in rows if r["controller"] == "policy"]
            made = sum(r["breaches_made"] for r in mine)
            stats[f"eval_{opp}_win"] = s["win"]
            stats[f"eval_{opp}_draw"] = s["draw"]
            stats[f"eval_{opp}_role_spec"] = _mean([r["role_specialization"] for r in mine])
            stats[f"eval_{opp}_breach_use"] = sum(r["breaches_used"] for r in mine) / made if made else float("nan")
            if opp == "heuristic":
                hm.save(os.path.join(self.out, "heatmaps", self.label, f"vs_{opp}_{tr.samples:012d}.npz"),
                        samples=tr.samples)
        return stats

    def replay(self):
        tr = self.tr
        path = os.path.join(self.out, "replays", self.label, f"{tr.samples:012d}_vs_heuristic.gif")
        record_round(self.replay_env, policy_controller(tr.model), scripted_controller("heuristic"), path)

    # ------------------------------------------------------------------ logging
    def _log(self, stats):
        row = {k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in stats.items()}
        if self._writer is None:
            fields = list(stats.keys())
            if os.path.exists(self.log_csv) and os.path.getsize(self.log_csv) > 0:
                with open(self.log_csv, newline="") as f:
                    fields = next(csv.reader(f))
                self._log_file = open(self.log_csv, "a", newline="")
                self._writer = csv.DictWriter(self._log_file, fieldnames=fields, extrasaction="ignore")
            else:
                self._log_file = open(self.log_csv, "w", newline="")
                self._writer = csv.DictWriter(self._log_file, fieldnames=fields, extrasaction="ignore")
                self._writer.writeheader()
        self._writer.writerow(row)
        self._log_file.flush()

    def _print(self, s):
        if not self.verbose:
            return
        ev = ""
        if s["eval_heuristic_win"] == s["eval_heuristic_win"]:
            ev = f" | vs heur {s['eval_heuristic_win']:.2f} vs split {s['eval_split_win']:.2f}"
        print(f"[{self.label}] it {s['iteration']:5d} | {s['samples']/1e6:9.1f}M | {s['sps']:9,.0f} sps | "
              f"ent {s['entropy']:.3f} kl {s['approx_kl']:.4f} ev {s['explained_var']:.2f} | "
              f"len {s['round_steps']:.0f} side0 {s['side0_win']:.2f} | pool {s['win_vs_pool']:.2f}{ev}",
              flush=True)

    # ------------------------------------------------------------------ main loop
    def run_until(self, target_samples):
        tr, tcfg = self.tr, self.tr.cfg
        try:
            while tr.samples < target_samples:
                try:
                    stats = tr.train_iteration()
                except FloatingPointError:
                    self.save_debug("nan")
                    raise
                it = tr.iteration
                if tcfg.check_every and it % tcfg.check_every == 0:
                    errs = tr.env.check_invariants()
                    if errs:
                        self.save_debug("invariant")
                        raise RuntimeError(f"simulator invariant violated at iteration {it}: {errs}")
                ev = {k: float("nan") for k in self.eval_keys}
                if self.eval_env is not None and (it % tcfg.eval_every == 0 or it == 1):
                    ev.update(self.evaluate())
                stats.update(ev)
                if self.replay_env is not None and (it % tcfg.replay_every == 0 or it == 1):
                    self.replay()
                self._log(stats)
                self._print(stats)
                if it % tcfg.save_every == 0:
                    self.save(tag=tr.samples)
        except KeyboardInterrupt:
            print("interrupted, saving", flush=True)
            self.save(tag=tr.samples)
            raise
        self.save(tag=tr.samples)

    def close(self):
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            self._writer = None
