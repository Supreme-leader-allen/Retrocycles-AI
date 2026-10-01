"""
Reward function. Everything except ``death`` is TEAM-SHARED: every cycle on a
team receives the same number, which is what makes this a cooperative credit
assignment problem rather than 7 independent agents.

    win           +win to the winning team, -win to the losing team, 0 on a draw
    conquest      (shaping) progress gained on the enemy zone minus progress lost on your own
    kill          (shaping) per enemy death minus per ally death
    death         (shaping, individual) penalty to the cycle that crashed

Shaping terms are multiplied by ``scale`` (annealed from 1 -> 0 by the trainer)
so the final policy is optimised for winning, not for the shaping.
"""
from dataclasses import dataclass, asdict

import torch


@dataclass
class RewardConfig:
    win: float = 1.0
    conquest: float = 1.0
    kill: float = 0.1
    death: float = 0.1
    anneal_samples: float = 4e8   # agent-steps until shaping reaches 0; <=0 disables annealing

    def scale(self, samples: float) -> float:
        if self.anneal_samples <= 0:
            return 1.0
        return max(0.0, 1.0 - samples / self.anneal_samples)

    def to_dict(self):
        return asdict(self)


def compute_rewards(info: dict, team: torch.Tensor, T: int, rcfg: RewardConfig, scale: float):
    """
    Returns (reward, parts): reward is (N, A); parts maps each component name
    ("win", "conquest", "kill", "death") to its (N, A) contribution, already
    scaled, so parts sum to reward. Logged by the trainer for debugging.
    """
    N = info["done"].shape[0]
    done = info["done"]
    winner = info["winner"]

    win_t = torch.zeros((N, 2), device=done.device)
    conq_t = torch.zeros((N, 2), device=done.device)
    kill_t = torch.zeros((N, 2), device=done.device)
    dprog = info["progress"] - info["progress_before"]                 # (N, 2) progress on zone k
    deaths = info["died"].view(N, 2, T).sum(2).float()                  # (N, 2) deaths on team k
    for k in range(2):
        won = done & (winner == k)
        lost = done & (winner == 1 - k)
        win_t[:, k] = rcfg.win * (won.float() - lost.float())
        conq_t[:, k] = scale * rcfg.conquest * (dprog[:, 1 - k] - dprog[:, k])
        kill_t[:, k] = scale * rcfg.kill * (deaths[:, 1 - k] - deaths[:, k])

    parts = {
        "win": win_t[:, team],
        "conquest": conq_t[:, team],
        "kill": kill_t[:, team],
        "death": -scale * rcfg.death * info["died"].float(),
    }
    r = parts["win"] + parts["conquest"] + parts["kill"] + parts["death"]
    return r, parts
