"""Residual-state comparator with continuous-time threshold crossings (Sec. 3.3).

Each pixel keeps a reference level S_ref. Within a window (t0, t0 + dt] the
driving signal is modelled locally as

    linear:     S(d) = S0 + S0_dot d                                  (Eq. 16)
    quadratic:  S(d) = S0 + S0_dot d + a d^2, fitted to S0, S0_dot, S1

and crossing times of S_ref + C+ / S_ref - C- are solved in closed form
(Eq. 17). Up to ``n_max`` events per pixel per window are emitted; after each
event S_ref moves by one threshold and the search continues on the remaining
interval. Everything is vectorised over pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ComparatorConfig:
    c_pos: float = 0.2
    c_neg: float = 0.2
    c_sigma: float = 0.0  # per-pixel threshold jitter (std), 0 = off
    c_min: float = 0.01
    n_max: int = 8  # max events per pixel per window ("No multi-crossing" ablation: 1)
    model: str = "quadratic"  # "linear" | "quadratic"
    leak_rate: float = 0.0  # lambda_leak, log units per second; 0 = off
    catchup: str = "emit"  # what to do if a window starts beyond a threshold: "emit" | "reset"
    seed: int = 0


class Comparator:
    def __init__(self, cfg: ComparatorConfig, shape, device="cpu"):
        self.cfg = cfg
        H, W = shape
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        if cfg.c_sigma > 0:
            cp = cfg.c_pos + cfg.c_sigma * torch.randn(H, W, generator=g)
            cn = cfg.c_neg + cfg.c_sigma * torch.randn(H, W, generator=g)
        else:
            cp = torch.full((H, W), cfg.c_pos)
            cn = torch.full((H, W), cfg.c_neg)
        self.c_pos = cp.clamp(min=cfg.c_min).to(device)
        self.c_neg = cn.clamp(min=cfg.c_min).to(device)
        self.s_ref = None
        self.device = device
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        self.xs, self.ys = xs.reshape(-1), ys.reshape(-1)

    def reset(self, S0: torch.Tensor):
        self.s_ref = S0.clone()

    # ------------------------------------------------------------------
    @staticmethod
    def _first_root(level, S0, Sd, a, lo, hi, quadratic: bool):
        """Smallest root of S(d) = level with lo < d <= hi, else +inf (all tensors flat)."""
        inf = torch.full_like(S0, float("inf"))
        c0 = S0 - level
        if not quadratic:
            d = -c0 / torch.where(Sd != 0, Sd, torch.ones_like(Sd))
            ok = (Sd != 0) & (d > lo) & (d <= hi)
            return torch.where(ok, d, inf)
        # a d^2 + Sd d + c0 = 0; fall back to linear where |a| is negligible
        lin = a.abs() * hi * hi < 1e-7 * (Sd.abs() * hi + c0.abs() + 1e-12)
        disc = Sd * Sd - 4 * a * c0
        sq = torch.sqrt(torch.clamp(disc, min=0))
        # numerically stable pair of roots
        qq = -0.5 * (Sd + torch.where(Sd >= 0, sq, -sq))
        a_safe = torch.where(a != 0, a, torch.ones_like(a))
        q_safe = torch.where(qq != 0, qq, torch.ones_like(qq))
        r1 = torch.where(qq != 0, qq / a_safe, inf)
        r2 = torch.where(qq != 0, c0 / q_safe, torch.where(a != 0, torch.zeros_like(a), inf))
        r1 = torch.where((disc >= 0) & (r1 > lo) & (r1 <= hi), r1, inf)
        r2 = torch.where((disc >= 0) & (r2 > lo) & (r2 <= hi), r2, inf)
        rq = torch.minimum(r1, r2)
        dl = -c0 / torch.where(Sd != 0, Sd, torch.ones_like(Sd))
        rl = torch.where((Sd != 0) & (dl > lo) & (dl <= hi), dl, inf)
        return torch.where(lin, rl, rq)

    def step(self, t0: float, dt: float, S0, S0_dot, S1=None, emit_mask=None):
        """Process the window (t0, t0 + dt].

        Args:
            S0, S0_dot: signal and its analytic slope at t0, (H, W).
            S1: signal at t0 + dt (needed for the quadratic model).
            emit_mask: (H, W) bool; False suppresses events (mixed-visibility
                masking, Sec. 3.4). Suppressed pixels re-anchor S_ref to S1.
        Returns:
            dict of 1-D tensors t (float64 seconds), x, y, p (+1/-1)
        """
        cfg = self.cfg
        if self.s_ref is None:
            self.reset(S0)
        if cfg.leak_rate > 0:
            self.s_ref = self.s_ref - cfg.leak_rate * dt
        S0f, Sdf = S0.reshape(-1), S0_dot.reshape(-1)
        ref = self.s_ref.reshape(-1).clone()
        cp, cn = self.c_pos.reshape(-1), self.c_neg.reshape(-1)
        quadratic = cfg.model == "quadratic" and S1 is not None
        if quadratic:
            a = (S1.reshape(-1) - S0f - Sdf * dt) / (dt * dt)
        else:
            a = torch.zeros_like(S0f)
        allowed = torch.ones_like(S0f, dtype=torch.bool) if emit_mask is None else emit_mask.reshape(-1)

        ts, xs, ys, ps = [], [], [], []
        n_left = torch.full_like(S0f, cfg.n_max, dtype=torch.int32)

        # catch-up: window starts already beyond a threshold (linear-model drift, n_max overflow)
        up = torch.floor((S0f - ref) / cp).clamp(min=0)
        dn = torch.floor((ref - S0f) / cn).clamp(min=0)
        if cfg.catchup == "emit":
            for sign, cnt, c in ((1, up, cp), (-1, dn, cn)):
                cnt = torch.minimum(cnt, n_left.float()) * allowed
                k = cnt.long()
                if k.sum() > 0:
                    idx = torch.repeat_interleave(torch.arange(len(k), device=k.device), k)
                    ts.append(torch.zeros(len(idx), device=idx.device))
                    xs.append(self.xs[idx]); ys.append(self.ys[idx])
                    ps.append(torch.full((len(idx),), sign, dtype=torch.int8, device=idx.device))
                    n_left = n_left - k.int()
                ref = ref + sign * cnt * c
        # anything still beyond (overflow or masked) is re-anchored
        ref = torch.where((S0f - ref >= cp) | (ref - S0f >= cn), S0f, ref)

        lo = torch.zeros_like(S0f)
        hi = float(dt)
        active = allowed & (n_left > 0)
        for _ in range(cfg.n_max):
            if not bool(active.any()):
                break
            idx = torch.nonzero(active).squeeze(-1)
            d_pos = self._first_root(ref[idx] + cp[idx], S0f[idx], Sdf[idx], a[idx], lo[idx], hi, quadratic)
            d_neg = self._first_root(ref[idx] - cn[idx], S0f[idx], Sdf[idx], a[idx], lo[idx], hi, quadratic)
            d = torch.minimum(d_pos, d_neg)
            fire = torch.isfinite(d)
            fi = idx[fire]
            if len(fi) == 0:
                break
            pol = torch.where(d_pos[fire] <= d_neg[fire], 1, -1).to(torch.int8)
            ts.append(d[fire]); xs.append(self.xs[fi]); ys.append(self.ys[fi]); ps.append(pol)
            ref[fi] = ref[fi] + torch.where(pol > 0, cp[fi], -cn[fi])
            lo[fi] = d[fire]
            n_left[fi] -= 1
            new_active = torch.zeros_like(active)
            new_active[fi] = n_left[fi] > 0
            active = new_active

        if emit_mask is not None and S1 is not None:
            ref = torch.where(allowed, ref, S1.reshape(-1))
        self.s_ref = ref.reshape(self.s_ref.shape)

        if ts:
            t = torch.cat(ts).double() + t0
            out = dict(t=t, x=torch.cat(xs), y=torch.cat(ys), p=torch.cat(ps))
            order = torch.argsort(out["t"], stable=True)
            return {k: v[order] for k, v in out.items()}
        e = torch.zeros(0, device=S0.device)
        return dict(t=e.double(), x=e.long(), y=e.long(), p=e.to(torch.int8))
