"""Derivative-conditioned finite-bandwidth readout (Sec. 3.5, Eq. 19).

Tiles keep FIFO queues; a deterministic scanner visits the tiles in raster
order once per tick and releases at most K_m events from tile m. Each tile's
throughput, drop probability and jitter degrade with its activity score psi_m:

    K_m      = floor(K0 (1 - beta_K psi_m))      (at least 1)
    p_drop,m = p0 + eta psi_m
    sigma_t,m = sigma0 + gamma_sigma psi_m

psi_m in [0, 1] is the tile mean of three renderer cues, each scaled to [0, 1]:
event-rate proxy |L_dot| / C * dt_base, transmittance rate |alpha_dot| * dt_base,
and contributor count c / c_ref. It is recorded once per simulation window.
None of the constants are given in the paper; defaults are ours.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:  # optional acceleration
    import numba

    _njit = numba.njit(cache=True)
except Exception:  # pragma: no cover
    numba = None
    _njit = lambda f: f  # noqa: E731


@dataclass
class ReadoutConfig:
    enabled: bool = True
    tile: int = 32  # pixels
    tick: float = 1e-6  # seconds per scanner sweep
    K0: int = 4
    p0: float = 0.0
    sigma0: float = 1e-6
    beta_K: float = 0.5
    eta: float = 0.05
    gamma_sigma: float = 10e-6
    c_ref: float = 32.0
    seed: int = 0


def tile_activity(frame, contrast: float, dt_base: float, tile: int, c_ref: float, use_alpha: bool = True) -> torch.Tensor:
    """psi_m for every tile, shape (n_tiles_y, n_tiles_x)."""
    cues = []
    if frame.L_dot is not None:
        cues.append((frame.L_dot.abs() / contrast * dt_base).clamp(0, 1))
    if use_alpha and frame.alpha_dot is not None:
        cues.append((frame.alpha_dot.abs() * dt_base).clamp(0, 1))
    cues.append((frame.count / c_ref).clamp(0, 1))
    x = torch.stack(cues).mean(0)
    H, W = x.shape
    ph, pw = (-H) % tile, (-W) % tile
    x = torch.nn.functional.pad(x[None, None], (0, pw, 0, ph))
    return torch.nn.functional.avg_pool2d(x, tile)[0, 0]


@_njit
def _arbitrate(tile_id, tick_in, K, n_tiles):
    """FIFO with capacity K[i] per tick per tile. Events must be time-sorted."""
    n = len(tile_id)
    out = np.empty(n, np.int64)
    last = np.full(n_tiles, -(2**62), np.int64)
    used = np.zeros(n_tiles, np.int64)
    for i in range(n):
        m = tile_id[i]
        c = tick_in[i]
        if c > last[m]:
            last[m] = c
            used[m] = 0
        if used[m] >= K[i]:
            last[m] += 1
            used[m] = 0
        used[m] += 1
        out[i] = last[m]
    return out


class TileArbiter:
    def __init__(self, cfg: ReadoutConfig, shape):
        self.cfg = cfg
        H, W = shape
        self.nty = (H + cfg.tile - 1) // cfg.tile
        self.ntx = (W + cfg.tile - 1) // cfg.tile
        self.windows = []  # (t_start, t_end, psi (nty, ntx) numpy)

    def record(self, t0: float, t1: float, psi: torch.Tensor):
        self.windows.append((t0, t1, psi.detach().float().cpu().numpy()))

    def __call__(self, ev: dict) -> dict:
        """ev: numpy dict t (float64 s), x, y, p sorted by t. Returns a new dict."""
        cfg = self.cfg
        n = len(ev["t"])
        if n == 0 or not cfg.enabled:
            return ev
        rng = np.random.default_rng(cfg.seed)
        t = ev["t"]
        tx = ev["x"] // cfg.tile
        ty = ev["y"] // cfg.tile
        tile_id = (ty * self.ntx + tx).astype(np.int64)
        # psi per event from the window it was generated in
        starts = np.array([w[0] for w in self.windows])
        wi = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(self.windows) - 1)
        psi_stack = np.stack([w[2] for w in self.windows])  # (nw, nty, ntx)
        psi = psi_stack[wi, ty, tx]
        K = np.maximum(1, np.floor(cfg.K0 * (1 - cfg.beta_K * psi))).astype(np.int64)
        p_drop = np.clip(cfg.p0 + cfg.eta * psi, 0, 1)
        sig = cfg.sigma0 + cfg.gamma_sigma * psi
        keep = rng.random(n) >= p_drop
        t, tile_id, K, sig = t[keep], tile_id[keep], K[keep], sig[keep]
        others = {k: v[keep] for k, v in ev.items() if k != "t"}
        t_ref = t[0] if len(t) else 0.0
        tick_in = np.ceil((t - t_ref) / cfg.tick).astype(np.int64)
        tick_out = _arbitrate(tile_id, tick_in, K, self.ntx * self.nty)
        n_tiles = self.ntx * self.nty
        t_out = t_ref + (tick_out + tile_id / n_tiles) * cfg.tick + rng.normal(0, 1, len(t)) * sig
        t_out = np.maximum(t_out, t)  # causal: never earlier than the crossing
        order = np.argsort(t_out, kind="stable")
        out = {"t": t_out[order]}
        out.update({k: v[order] for k, v in others.items()})
        return out
