"""Deformable scene interface: maps time to raw per-Gaussian parameters (Eq. 6)."""
from __future__ import annotations

import math

import torch


class DeformableScene:
    """Theta(t) = (mu, s_raw, q_raw, sigma_raw, sh) for every Gaussian.

    ``params(times)`` takes a per-Gaussian time vector ``times`` of shape (P,)
    (all entries equal in normal use). Taking a vector rather than a scalar lets
    the reverse-mode fallback recover per-Gaussian time derivatives exactly,
    because each Gaussian's parameters depend only on its own time entry.

    Times passed in are *scene* times (whatever the scene was trained with, for
    4DGS a normalised [0, 1] value); the simulator converts from seconds.
    """

    sh_degree: int = 0
    num_gaussians: int = 0

    def params(self, times: torch.Tensor) -> dict:
        raise NotImplementedError

    def to(self, device):
        return self


class ToyScene(DeformableScene):
    """Small analytic scene for tests and demos.

    Each Gaussian moves on a straight line with constant velocity, and can have
    a sinusoidally varying log-scale and opacity. Everything is closed form, so
    exact time derivatives are known.
    """

    def __init__(self, means, velocities, scales_raw, quats, opacity_raw, colors, sh_degree=0,
                 scale_freq=0.0, opacity_freq=0.0, device="cpu"):
        self.device = device
        f = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        self.means = f(means)
        self.vel = f(velocities)
        self.scales_raw = f(scales_raw)
        self.quats = f(quats)
        self.opacity_raw = f(opacity_raw).reshape(-1, 1)
        colors = f(colors)
        K = (sh_degree + 1) ** 2
        sh = torch.zeros(len(colors), K, 3, device=device)
        sh[:, 0] = (colors - 0.5) / 0.28209479177387814
        self.shs = sh
        self.sh_degree = sh_degree
        self.num_gaussians = len(colors)
        self.scale_freq = scale_freq
        self.opacity_freq = opacity_freq

    @classmethod
    def random(cls, n=64, seed=0, depth=(3.0, 6.0), spread=1.5, speed=0.5, device="cpu", **kw):
        g = torch.Generator().manual_seed(seed)
        u = lambda *s: torch.rand(*s, generator=g)
        means = torch.stack([(u(n) - 0.5) * 2 * spread, (u(n) - 0.5) * 2 * spread, depth[0] + u(n) * (depth[1] - depth[0])], -1)
        vel = (u(n, 3) - 0.5) * 2 * speed
        scales_raw = torch.log(0.05 + 0.25 * u(n, 3))
        quats = torch.randn(n, 4, generator=g)
        opacity_raw = torch.logit(0.3 + 0.65 * u(n))
        colors = 0.1 + 0.8 * u(n, 3)
        return cls(means, vel, scales_raw, quats, opacity_raw, colors, device=device, **kw)

    def params(self, times):
        t = times.reshape(-1, 1)
        mu = self.means + self.vel * t
        s_raw = self.scales_raw + 0.2 * torch.sin(2 * math.pi * self.scale_freq * t) if self.scale_freq else self.scales_raw + 0 * t
        o_raw = self.opacity_raw + 1.0 * torch.sin(2 * math.pi * self.opacity_freq * t) if self.opacity_freq else self.opacity_raw + 0 * t
        return dict(means=mu, scales_raw=s_raw, quats_raw=self.quats + 0 * t, opacity_raw=o_raw.reshape(-1),
                    shs=self.shs + 0 * t.unsqueeze(-1))
