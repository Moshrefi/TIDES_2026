"""Risk-guided adaptive time stepping (Sec. 3.4, Eq. 18).

    r(u,t) = clip(w_b (1 - norm(b)) + w_c norm(c) + w_a norm(|alpha_dot|))

The renderer is queried for the whole image at each sub-pose, so the step is
global: dt = dt_base * kappa(r_agg), with r_agg a high quantile of the per-pixel
risk, and kappa(r) = 1 / (1 + k r) in (0, 1].

The paper does not give norm(), kappa(), the weights or the aggregation; the
choices below are ours and are all configurable. norm() uses fixed physical
scales rather than per-frame statistics, so a calm frame really gets a calm
(large) step:
    norm(b)          = b                         (b = sum_k w_k is already in [0, 1])
    norm(c)          = min(c / c_ref, 1)
    norm(|alpha_dot|) = min(|alpha_dot| * dt_base, 1)   (opacity change over one base step)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AdaptiveConfig:
    enabled: bool = False
    base_fraction: float = 0.25  # dt_base = frame interval * base_fraction (adaptive mode)
    fixed_fraction: float = 0.1  # dt = frame interval * fixed_fraction (fixed 10x mode)
    w_b: float = 1.0 / 3
    w_c: float = 1.0 / 3
    w_a: float = 1.0 / 3
    c_ref: float = 32.0
    k: float = 7.0  # kappa(1) = 1/8 -> up to 8x refinement
    quantile: float = 0.99
    use_alpha_cue: bool = True  # Table 4 "No alpha_dot cue"
    use_risk: bool = True  # Table 4 "No risk-gated refinement"
    b_min: float = 0.05
    use_mask: bool = True  # Table 4 "No mixed-visibility masking"
    # basic rate controller (on top of which the risk shrink acts): keep the
    # predicted number of crossings per step at most `max_crossings` for the
    # `rate_quantile` busiest pixel. Table 4 "Fixed step size (no controller)"
    # turns both this and the risk shrink off.
    rate_control: bool = True
    max_crossings: float = 4.0
    rate_quantile: float = 0.99
    contrast: float = 0.2
    min_fraction: float = 1.0 / 64  # dt never below frame interval * min_fraction


def risk_map(frame, cfg: AdaptiveConfig, dt_base: float) -> torch.Tensor:
    b = frame.coverage.clamp(0, 1)
    c = (frame.count / cfg.c_ref).clamp(0, 1)
    r = cfg.w_b * (1 - b) + cfg.w_c * c
    if cfg.use_alpha_cue and frame.alpha_dot is not None:
        r = r + cfg.w_a * (frame.alpha_dot.abs() * dt_base).clamp(0, 1)
    return r.clamp(0, 1)


def kappa(r: float, k: float) -> float:
    return 1.0 / (1.0 + k * r)


class StepController:
    def __init__(self, cfg: AdaptiveConfig, frame_interval: float):
        self.cfg = cfg
        self.frame_interval = frame_interval
        if cfg.enabled:
            self.dt_base = frame_interval * cfg.base_fraction
        else:
            self.dt_base = frame_interval * cfg.fixed_fraction

    @staticmethod
    def _quantile(x, q):
        x = x.reshape(-1)
        if x.numel() == 0:
            return 0.0
        if x.numel() > 2_000_000:
            x = x[torch.randperm(x.numel(), device=x.device)[:2_000_000]]
        return float(torch.quantile(x.float(), q))

    def next_dt(self, frame) -> float:
        cfg = self.cfg
        if not cfg.enabled:
            return self.dt_base
        dt = self.dt_base
        emit = frame.coverage >= cfg.b_min if cfg.use_mask else torch.ones_like(frame.coverage, dtype=torch.bool)
        if cfg.rate_control and frame.L_dot is not None:
            rate = self._quantile(frame.L_dot.abs()[emit], cfg.rate_quantile)
            if rate > 0:
                dt = min(dt, cfg.max_crossings * cfg.contrast / rate)
        if cfg.use_risk:
            # only pixels that can emit count (pure background is masked anyway)
            r = risk_map(frame, cfg, self.dt_base)[emit]
            dt = dt * kappa(self._quantile(r, cfg.quantile), cfg.k)
        return max(dt, self.frame_interval * cfg.min_fraction)

    def emit_mask(self, frame):
        if not self.cfg.use_mask:
            return None
        return frame.coverage >= self.cfg.b_min
