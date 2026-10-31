"""Optional sensor front-end F (Sec. 3.3). Identity by default.

Blocks (applied in this order when enabled):
  * optical PSF blur  - Gaussian blur of L and L_dot (linear, so it commutes with d/dt)
  * centre-surround   - S = L - k * blur_s(L), likewise for the slope
  * photoreceptor low-pass - first-order filter  tau dS/dt = L - S, so S_dot = (L - S) / tau

Leak is handled in the comparator (reference drift).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SensorConfig:
    psf_sigma: float = 0.0  # pixels, 0 = off
    surround_sigma: float = 0.0  # pixels, 0 = off
    surround_gain: float = 0.0
    lowpass_tau: float = 0.0  # seconds, 0 = off


def _gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    y = x[None, None]
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return y[0, 0]


class SensorFrontEnd:
    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg
        self.state = None  # photoreceptor state
        self.t_state = None

    @property
    def identity(self):
        c = self.cfg
        return c.psf_sigma <= 0 and c.surround_sigma <= 0 and c.lowpass_tau <= 0

    def __call__(self, t: float, L: torch.Tensor, L_dot: torch.Tensor):
        c = self.cfg
        S, Sd = L, L_dot
        if c.psf_sigma > 0:
            S, Sd = _gauss_blur(S, c.psf_sigma), _gauss_blur(Sd, c.psf_sigma)
        if c.surround_sigma > 0 and c.surround_gain > 0:
            S = S - c.surround_gain * _gauss_blur(S, c.surround_sigma)
            Sd = Sd - c.surround_gain * _gauss_blur(Sd, c.surround_sigma)
        if c.lowpass_tau > 0:
            if self.state is None:
                self.state = S.clone()
            else:
                # exact update for an input that is linear over the step
                dt = t - self.t_state
                e = math.exp(-dt / c.lowpass_tau)
                self.state = S + (self.state - S) * e
            self.t_state = t
            Sd = (S - self.state) / c.lowpass_tau
            S = self.state
        return S, Sd
