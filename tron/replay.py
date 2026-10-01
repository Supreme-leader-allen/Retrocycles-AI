"""
Replay recording (GIF) and position heatmaps.

Colours: team 0 = blue (defends the bottom zone), team 1 = orange (top zone).
Bright squares are cycle heads, darker cells are trails, tinted circles are the
zones, and the bars at the top show each zone's conquest progress.
"""
import os

import numpy as np
import torch

TEAM_TRAIL = np.array([[40, 110, 220], [230, 120, 30]], dtype=np.uint8)
TEAM_HEAD = np.array([[150, 220, 255], [255, 220, 120]], dtype=np.uint8)
ZONE_TINT = np.array([[18, 30, 60], [60, 35, 15]], dtype=np.uint8)
RIM = np.array([90, 90, 90], dtype=np.uint8)
BG = np.array([8, 8, 12], dtype=np.uint8)


def frame(env, n=0, scale=6):
    P, S, T = env.P, env.S, env.T
    owner = env.owner[n, P - 1:P + S + 1, P - 1:P + S + 1].cpu().numpy().astype(np.int64)
    zone = env.zone_grid[P - 1:P + S + 1, P - 1:P + S + 1].cpu().numpy().astype(np.int64)
    img = np.empty(owner.shape + (3,), dtype=np.uint8)
    img[:] = BG
    for k in range(2):
        img[zone == k] = ZONE_TINT[k]
    img[owner == -2] = RIM
    trail = owner >= 0
    img[trail] = TEAM_TRAIL[owner[trail] // T]
    for a in range(env.A):
        if env.alive[n, a]:
            y, x = env.pos[n, a].tolist()
            img[y - P + 1, x - P + 1] = TEAM_HEAD[a // T]
    img = img.repeat(scale, 0).repeat(scale, 1)
    bar = np.zeros((6, img.shape[1], 3), dtype=np.uint8)
    w = img.shape[1] // 2
    for k in range(2):  # left half = team 0's zone, right half = team 1's zone
        p = float(env.progress[n, k])
        bar[:, k * w:k * w + int(p * (w - 2))] = TEAM_TRAIL[1 - k]
    return np.concatenate([bar, img], 0)


def record_round(env, ctrl_team0, ctrl_team1, gif_path=None, scale=6, fps=20):
    """Play one round in env 0 of ``env`` (other envs, if any, also step but are ignored).
    Returns (frames, info of the final step). Saves a GIF if ``gif_path`` is given."""
    from PIL import Image
    env.reset_done(torch.ones(env.N, dtype=torch.bool, device=env.device))
    team0 = (env.team == 0).unsqueeze(0)
    frames = [frame(env, 0, scale)]
    while True:
        codes, vec = env.observe()
        act = torch.where(team0, ctrl_team0(env, codes, vec), ctrl_team1(env, codes, vec))
        info = env.step(act)
        frames.append(frame(env, 0, scale))
        if info["done"][0]:
            break
        env.reset_done(info["done"])  # keep any other envs going
    if gif_path:
        os.makedirs(os.path.dirname(os.path.abspath(gif_path)), exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs += [imgs[-1]] * fps
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    return frames, info


class HeatmapRecorder:
    """Where a controller's cycles drive and die, in a canonical frame (own zone at the
    bottom, so team 1's positions are rotated 180 degrees). Use as play_rounds(on_step=...)."""

    def __init__(self, env):
        self.S = env.S
        self.visits = torch.zeros((env.S, env.S), device=env.device)
        self.deaths = torch.zeros((env.S, env.S), device=env.device)
        self.env = env

    def _canon(self, mask):
        env, S, P = self.env, self.S, self.env.P
        n_i, a_i = mask.nonzero(as_tuple=True)
        y = env.pos[n_i, a_i, 0] - P
        x = env.pos[n_i, a_i, 1] - P
        flip = env.team[a_i] == 1
        y = torch.where(flip, S - 1 - y, y)
        x = torch.where(flip, S - 1 - x, x)
        return y, x

    def __call__(self, env, info, controlled):
        one = None
        y, x = self._canon(info["alive"] & controlled)
        if y.numel():
            one = torch.ones_like(y, dtype=self.visits.dtype)
            self.visits.index_put_((y, x), one, accumulate=True)
        y, x = self._canon(info["died"] & controlled)
        if y.numel():
            self.deaths.index_put_((y, x), torch.ones_like(y, dtype=self.deaths.dtype), accumulate=True)

    def save(self, path, **meta):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, visits=self.visits.cpu().numpy(), deaths=self.deaths.cpu().numpy(),
                            zone_center=self.env.zone_center.cpu().numpy() - self.env.P,
                            zone_radius=self.env.cfg.zone_radius, **meta)
