"""
PPO with parameter sharing and self-play, running directly on the vectorized
env (no worker processes).

One iteration = ``rollout_len`` decisions in each of ``num_envs`` rounds for
all ``2*team_size`` cycles = rollout_len * num_envs * A samples ("agent-steps").

Self-play: both teams are driven by the current policy, and both teams'
experience is trained on. In a fraction ``pool_frac`` of the envs, team 1 is
instead a frozen snapshot of an earlier policy (opponent pool), which stops
the policy from overfitting to beating only its current self; those
snapshot-controlled cycles are excluded from training.

Scripted opponents: in a further fraction ``bot_frac`` of the envs, team 1 is
a scripted bot (alternating heuristic all-attack rushers and split-role
teams). Pure self-play never meets an all-out rush, so without this the policy
learned to leave its base empty (7v7 pilot: frac_defending ~0.05, lost 60% of
games to the rush bot by conquest). Bot-controlled cycles are not trained on.

Credit after death (``credit_after_death=True``, the default): a cycle's
episode is the whole ROUND, not its own life. After it crashes it keeps
receiving its team's reward (masked out of the policy loss, since a dead cycle
has no actions, but kept in the value/advantage chain), so actions that helped
the team win before dying, e.g. a sacrificial cut, still get credit. With it
off, each cycle's episode ends when it dies: the "selfish credit" ablation.
"""
import copy
import math
import random
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .model import Policy, amp, setup_cuda_speed
from .rewards import RewardConfig, compute_rewards
from .metrics import MetricsTracker
from .bots import heuristic_actions
from .env import FortressEnv

REWARD_PARTS = ("win", "conquest", "kill", "death")


@dataclass
class TrainConfig:
    num_envs: int = 1024
    rollout_len: int = 64
    total_samples: float = 1e9
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    epochs: int = 3
    minibatch: int = 32768
    max_grad_norm: float = 0.5
    hidden: int = 256
    credit_after_death: bool = True
    pool_frac: float = 0.25
    pool_every: int = 20          # iterations between snapshots added to the pool
    pool_size: int = 30
    bot_frac: float = 0.2         # fraction of envs where team 1 is a scripted bot (heuristic / split)
    save_every: int = 50          # iterations
    eval_every: int = 25          # iterations; 0 disables
    eval_envs: int = 64
    eval_rounds: int = 2          # rounds per eval env (so eval_envs * eval_rounds rounds)
    replay_every: int = 100       # iterations between saved replay GIFs; 0 disables
    check_every: int = 10         # iterations between simulator invariant checks; 0 disables
    amp: bool = True              # bfloat16 mixed precision on CUDA (ignored on CPU)
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


class PPOTrainer:
    def __init__(self, game_cfg, train_cfg: TrainConfig, reward_cfg: RewardConfig, device,
                 metrics_csv=None, run_label="run", log=print):
        self.gcfg, self.cfg, self.rcfg = game_cfg, train_cfg, reward_cfg
        self.device = torch.device(device)
        if self.device.type == "cuda":
            setup_cuda_speed()
        self.log = log
        torch.manual_seed(train_cfg.seed)
        random.seed(train_cfg.seed)
        self.env = FortressEnv(game_cfg, train_cfg.num_envs, device=self.device, seed=train_cfg.seed)
        env = self.env
        self.model = Policy(env.K, env.vec_dim, hidden=train_cfg.hidden).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)  # tensor-core friendly convs
        self.opt = torch.optim.Adam(self.model.parameters(), lr=train_cfg.lr, eps=1e-5)
        self.opp_model = copy.deepcopy(self.model).eval()
        self.pool = []                      # list of CPU state_dicts
        self.tracker = MetricsTracker(env, csv_path=metrics_csv, run_label=run_label)
        self.iteration = 0
        self.samples = 0
        self.obs = env.observe()

        N, A, T = env.N, env.A, env.T
        self.n_pool_envs = int(round(train_cfg.pool_frac * N)) if train_cfg.pool_frac > 0 else 0
        self.n_bot_envs = int(round(train_cfg.bot_frac * N)) if train_cfg.bot_frac > 0 else 0
        assert self.n_pool_envs + self.n_bot_envs <= N, "pool_frac + bot_frac > 1"
        # envs [n_pool, n_pool + n_bot): team 1 is a bot; even ones heuristic, odd ones split
        self.bot_lo, self.bot_hi = self.n_pool_envs, self.n_pool_envs + self.n_bot_envs
        self.bot_split = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.bot_split[self.bot_lo + 1:self.bot_hi:2] = True
        self.trainable = torch.ones((N, A), dtype=torch.bool, device=self.device)
        self.pool_active = False

    # ------------------------------------------------------------------ hyperparameters (PBT)
    def set_hparams(self, **hp):
        """Change hyperparameters on a live trainer: lr, ent_coef (TrainConfig) or any RewardConfig field."""
        for k, v in hp.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
            elif hasattr(self.rcfg, k):
                setattr(self.rcfg, k, v)
            else:
                raise KeyError(k)
        for g in self.opt.param_groups:
            g["lr"] = self.cfg.lr

    # ------------------------------------------------------------------ pool
    def _snapshot(self):
        sd = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
        self.pool.append(sd)
        if len(self.pool) > self.cfg.pool_size:
            self.pool.pop(0)

    def _pick_opponent(self):
        T = self.env.T
        M = self.n_pool_envs
        self.trainable.fill_(True)
        self.pool_active = M > 0 and len(self.pool) > 0
        for n in range(self.env.N):
            self.tracker.controllers[n] = ["policy", "policy"]
        if self.n_bot_envs:
            self.trainable[self.bot_lo:self.bot_hi, T:] = False
            for n in range(self.bot_lo, self.bot_hi):
                self.tracker.controllers[n] = ["policy", "split" if self.bot_split[n] else "heuristic"]
        if not self.pool_active:
            return
        self.opp_model.load_state_dict(random.choice(self.pool))
        self.trainable[:M, T:] = False
        for n in range(M):
            self.tracker.controllers[n] = ["policy", "pool"]

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def collect(self):
        cfg, env = self.cfg, self.env
        N, A, T, K, L = env.N, env.A, env.T, env.K, cfg.rollout_len
        dev = self.device
        scale = self.rcfg.scale(self.samples)
        buf = {
            "codes": torch.empty((L, N, A, K, K), dtype=torch.uint8, device=dev),
            "vec": torch.empty((L, N, A, env.vec_dim), dtype=torch.float32, device=dev),
            "act": torch.empty((L, N, A), dtype=torch.long, device=dev),
            "logp": torch.empty((L, N, A), device=dev),
            "val": torch.empty((L, N, A), device=dev),
            "rew": torch.empty((L, N, A), device=dev),
            "done": torch.empty((L, N, A), device=dev),
            "pmask": torch.empty((L, N, A), dtype=torch.bool, device=dev),
            "vmask": torch.empty((L, N, A), dtype=torch.bool, device=dev),
        }
        part_sum = {p: torch.zeros((), device=dev) for p in REWARD_PARTS}
        M = self.n_pool_envs
        self.model.eval()
        for t in range(L):
            codes, vec = self.obs
            with amp(dev, cfg.amp):
                logits, value = self.model(codes.view(N * A, K, K), vec.view(N * A, -1))
            dist = Categorical(logits=logits)
            act = dist.sample()
            logp = dist.log_prob(act).view(N, A)
            act = act.view(N, A)
            value = value.view(N, A)
            if self.pool_active:
                with amp(dev, cfg.amp):
                    ol, _ = self.opp_model(codes[:M, T:].reshape(M * T, K, K), vec[:M, T:].reshape(M * T, -1))
                act[:M, T:] = Categorical(logits=ol).sample().view(M, T)
            if self.n_bot_envs:
                lo, hi = self.bot_lo, self.bot_hi
                bot = env.delay_actions(heuristic_actions(env, split_roles=self.bot_split), "train_bots")
                act[lo:hi, T:] = bot[lo:hi, T:]

            alive0 = env.alive.clone()
            info = env.step(act)
            rew, parts = compute_rewards(info, env.team, T, self.rcfg, scale)
            self.tracker.samples = self.samples
            self.tracker.update(info)

            env_done = info["done"].unsqueeze(1).expand(N, A)
            if cfg.credit_after_death:
                done = env_done
                vmask = self.trainable.clone()
            else:
                done = env_done | ~info["alive"]
                rew = rew * alive0.float()
                vmask = alive0 & self.trainable
            pmask = alive0 & self.trainable
            for p in REWARD_PARTS:
                part_sum[p] += (parts[p].abs() * vmask).sum()   # magnitude: signed means cancel in self-play

            buf["codes"][t] = codes
            buf["vec"][t] = vec
            buf["act"][t] = act
            buf["logp"][t] = logp
            buf["val"][t] = value
            buf["rew"][t] = rew
            buf["done"][t] = done.float()
            buf["pmask"][t] = pmask
            buf["vmask"][t] = vmask

            env.reset_done(info["done"])
            self.obs = env.observe()

        codes, vec = self.obs
        with amp(dev, cfg.amp):
            _, next_val = self.model(codes.view(N * A, K, K), vec.view(N * A, -1))
        next_val = next_val.view(N, A)

        # GAE
        adv = torch.zeros_like(buf["rew"])
        last = torch.zeros((N, A), device=dev)
        for t in reversed(range(L)):
            nv = next_val if t == L - 1 else buf["val"][t + 1]
            nonterm = 1.0 - buf["done"][t]
            delta = buf["rew"][t] + cfg.gamma * nv * nonterm - buf["val"][t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
            adv[t] = last
        buf["adv"] = adv
        buf["ret"] = adv + buf["val"]
        self.samples += L * N * A

        # rollout statistics (for debugging / graphs)
        nv_ = buf["vmask"].sum().clamp(min=1)
        np_ = buf["pmask"].sum().clamp(min=1)
        acts = buf["act"][buf["pmask"]]
        counts = torch.bincount(acts, minlength=3).float() / max(acts.numel(), 1)
        stats = {f"rew_{p}": float(part_sum[p] / nv_) for p in REWARD_PARTS}
        stats.update({
            "act_straight": float(counts[0]), "act_left": float(counts[1]), "act_right": float(counts[2]),
            "alive_frac": float(np_ / buf["vmask"].numel()),
            "value_mean": float(buf["val"][buf["vmask"]].mean()),
            "return_mean": float(buf["ret"][buf["vmask"]].mean()),
            "adv_std_raw": float(buf["adv"][buf["pmask"]].std()) if int(np_) > 1 else float("nan"),
        })
        return buf, scale, stats

    # ------------------------------------------------------------------ update
    def update(self, buf):
        cfg = self.cfg
        K = self.env.K
        vmask = buf["vmask"].reshape(-1)
        idx = vmask.nonzero(as_tuple=True)[0]
        codes = buf["codes"].reshape(-1, K, K)[idx]
        vec = buf["vec"].reshape(-1, self.env.vec_dim)[idx]
        act = buf["act"].reshape(-1)[idx]
        old_logp = buf["logp"].reshape(-1)[idx]
        adv = buf["adv"].reshape(-1)[idx]
        ret = buf["ret"].reshape(-1)[idx]
        pmask = buf["pmask"].reshape(-1)[idx].float()

        pa = adv[pmask > 0]
        adv = (adv - pa.mean()) / (pa.std() + 1e-8)

        self.model.train()
        n = idx.numel()
        keys = ("pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "grad_norm")
        stats = {k: 0.0 for k in keys}
        count = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, cfg.minibatch):
                mb = perm[s:s + cfg.minibatch]
                with amp(self.device, cfg.amp):
                    logits, v = self.model(codes[mb], vec[mb])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act[mb])
                ent = dist.entropy()
                pm = pmask[mb]
                denom = pm.sum().clamp(min=1.0)
                ratio = torch.exp(logp - old_logp[mb])
                a = adv[mb]
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
                pg_loss = (pg * pm).sum() / denom
                ent_mean = (ent * pm).sum() / denom
                v_loss = 0.5 * ((v - ret[mb]) ** 2).mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_mean
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at iteration {self.iteration}: pg {pg_loss.item()} "
                        f"v {v_loss.item()} ent {ent_mean.item()}")
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    lr_ = logp - old_logp[mb]
                    kl = (((torch.exp(lr_) - 1) - lr_) * pm).sum() / denom
                    cf = (((ratio - 1).abs() > cfg.clip).float() * pm).sum() / denom
                stats["pg_loss"] += pg_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent_mean.item()
                stats["approx_kl"] += kl.item()
                stats["clipfrac"] += cf.item()
                stats["grad_norm"] += float(gn)
                count += 1
        for k in stats:
            stats[k] /= max(count, 1)
        with torch.no_grad():
            var_y = ret.var()
            stats["explained_var"] = float(1 - (ret - buf["val"].reshape(-1)[idx]).var() / (var_y + 1e-8))
            stats["param_norm"] = float(torch.sqrt(sum((p.detach() ** 2).sum() for p in self.model.parameters())))
        return stats

    # ------------------------------------------------------------------ one iteration
    def train_iteration(self):
        if self.cfg.pool_every > 0 and self.iteration % self.cfg.pool_every == 0:
            self._snapshot()
        self._pick_opponent()
        t0 = time.time()
        buf, scale, roll = self.collect()
        t1 = time.time()
        stats = self.update(buf)
        t2 = time.time()
        self.iteration += 1
        rows = self.tracker.flush()
        n = self.cfg.rollout_len * self.env.N * self.env.A
        stats.update(roll)
        stats.update({
            "iteration": self.iteration, "samples": self.samples,
            "sps": n / (t2 - t0), "collect_s": t1 - t0, "update_s": t2 - t1,
            "lr": self.cfg.lr, "ent_coef": self.cfg.ent_coef, "shaping_scale": scale,
            "mean_reward": float(buf["rew"][buf["vmask"]].mean()),
            "pool_size": len(self.pool),
            "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if self.device.type == "cuda" else float("nan"),
        })
        stats.update(summarize_rows(rows))
        return stats

    # ------------------------------------------------------------------ checkpointing
    def state(self):
        return {
            "model": self.model.state_dict(), "opt": self.opt.state_dict(),
            "model_kwargs": self.model.kwargs,
            "iteration": self.iteration, "samples": self.samples, "pool": self.pool,
            "game_cfg": self.gcfg.to_dict(), "train_cfg": self.cfg.to_dict(),
            "reward_cfg": self.rcfg.to_dict(),
        }

    def load(self, ckpt):
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.iteration = ckpt["iteration"]
        self.samples = ckpt["samples"]
        self.pool = ckpt.get("pool", [])
        for g in self.opt.param_groups:
            g["lr"] = self.cfg.lr


def summarize_rows(rows):
    """Per-iteration summary of the training rounds that finished (for the train log)."""
    sp = [r for r in rows if r["controller"] == "policy" and r["opponent"] == "policy"]
    vp = [r for r in rows if r["controller"] == "policy" and r["opponent"] == "pool"]
    out = {"rounds": len(rows) // 2}
    n = len(sp)
    out["round_steps"] = _mean([r["round_steps"] for r in sp])
    out["selfplay_draw"] = _mean([float(r["result"] == "draw") for r in sp])
    # side balance: in self-play both sides are the same policy, so team 0 should win ~50% of decisive
    # rounds. A persistent imbalance points to an asymmetry bug in the simulator or observations.
    dec0 = [r for r in sp if r["team"] == 0 and r["result"] != "draw"]
    out["side0_win"] = _mean([float(r["result"] == "win") for r in dec0])
    for reason in ("conquest", "elimination", "timeout"):
        out[f"end_{reason}"] = _mean([float(r["end_reason"] == reason) for r in sp if r["team"] == 0])
    for c in ("deaths_self", "friendly_fire_deaths", "deaths_enemy", "deaths_rim", "deaths_headon",
              "kills", "breaches_made", "breaches_used"):
        out[c] = _mean([r[c] for r in sp])
    for c in ("role_entropy", "role_specialization", "frac_defending", "frac_attacking",
              "multi_attack_rate", "undefended_rate", "avg_teammate_dist", "survival",
              "avg_speed", "turn_rate"):
        out[c] = _mean([r[c] for r in sp])
    out["win_vs_pool"] = _mean([float(r["result"] == "win") for r in vp])
    for bot in ("heuristic", "split"):
        vb = [r for r in rows if r["controller"] == "policy" and r["opponent"] == bot]
        out[f"train_win_vs_{bot}"] = _mean([float(r["result"] == "win") for r in vb])
    defending = [r["frac_defending"] for r in rows if r["controller"] == "policy" and r["opponent"] != "policy"]
    out["frac_defending_vs_opponents"] = _mean(defending)
    return out


def load_policy(path, device="cpu"):
    """Load just the network from a checkpoint (for evaluation / rendering)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Policy(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
