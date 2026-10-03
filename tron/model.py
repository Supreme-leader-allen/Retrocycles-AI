"""
Shared policy/value network. Every cycle on both teams uses the same weights
(parameter sharing); the observation is egocentric, so "my team" vs "the
enemy" is always from the viewer's point of view.

    codes (K x K cell codes) -> embedding -> 3 conv layers ─┐
    vec (self + all other cycles) -> MLP ───────────────────┴-> MLP -> policy logits (3)
                                                                   -> value (1)
"""
import contextlib

import torch
import torch.nn as nn

from .env import NUM_CELL_CODES


def amp(device, enabled=True):
    """bfloat16 autocast on CUDA (tensor cores), a no-op on CPU."""
    device = torch.device(device)
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def setup_cuda_speed():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _init(layer, gain=2 ** 0.5):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Policy(nn.Module):
    def __init__(self, crop: int, vec_dim: int, hidden: int = 256, emb: int = 8):
        super().__init__()
        self.kwargs = dict(crop=crop, vec_dim=vec_dim, hidden=hidden, emb=emb)
        self.emb = nn.Embedding(NUM_CELL_CODES, emb)
        self.conv = nn.Sequential(
            _init(nn.Conv2d(emb, 32, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(64, 64, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, emb, crop, crop)).shape[1]
        self.vec = nn.Sequential(_init(nn.Linear(vec_dim, hidden)), nn.ReLU())
        self.trunk = nn.Sequential(
            _init(nn.Linear(conv_out + hidden, 2 * hidden)), nn.ReLU(),
            _init(nn.Linear(2 * hidden, hidden)), nn.ReLU(),
        )
        self.pi = _init(nn.Linear(hidden, 3), gain=0.01)
        self.v = _init(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, codes, vec):
        # (B, K, K, E) permuted to (B, E, K, K) is already channels_last in memory
        x = self.emb(codes.long()).permute(0, 3, 1, 2)
        h = torch.cat([self.conv(x), self.vec(vec)], dim=1)
        h = self.trunk(h)
        # heads always in float32, even under bfloat16 autocast: PPO's probability
        # ratios and the value regression need the precision
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            return self.pi(h), self.v(h).squeeze(-1)
