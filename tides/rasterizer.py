"""Forward-mode time-derivative Gaussian rasterizer (Sec. 3.1-3.2, Eq. 1-15).

Pure-torch, tile-based implementation that follows the reference 3DGS CUDA
rasterizer's decisions exactly:

* 16x16 tiles; a Gaussian is listed in every tile its 3-sigma rectangle touches
* per-tile front-to-back order by view-space depth
* ``power > 0`` skipped, ``alpha = min(0.99, sigma * g)``, ``alpha < 1/255`` skipped
* early termination when ``T * (1 - alpha) < 1e-4`` (that Gaussian excluded)

Alongside the primal colour and transmittance it propagates the time-tangents
through the same compositing chain (Eq. 10-14):

    p_dot   = -1/2 d^T Q_dot d - d_dot^T Q d                 (Eq. 10; here d = u_k - u)
    tau_dot = sigma_dot g + sigma g p_dot, 0 if saturated    (Eq. 11)
    w_dot   = T_dot_{k-1} tau + T_{k-1} tau_dot              (Eq. 12)
    T_dot_k = T_dot_{k-1} (1 - tau) - T_{k-1} tau_dot        (Eq. 13)
    I_dot   = sum(w_dot c + w c_dot) + T_dot_K I_bg          (Eq. 14)

Inside a chunk of K Gaussians the recursions are evaluated in closed form:
``T_k = T_0 prod(1 - tau_j)`` and ``T_dot_k / T_k = T_dot_0/T_0 - sum tau_dot_j / (1 - tau_j)``
(exact, since 1 - tau >= 0.01).

It also returns the diagnostics used by the adaptive controller and readout
model (Sec. 3.4): contributor count c(u,t) and coverage b(u,t) = sum_k w_k.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

TAU_MAX = 0.99
ALPHA_MIN = 1.0 / 255.0
T_MIN = 1e-4
BLOCK = 16


@dataclass
class RasterOutput:
    rgb: torch.Tensor  # (H, W, 3)
    rgb_dot: torch.Tensor | None  # (H, W, 3)
    trans: torch.Tensor  # (H, W) final transmittance T_K
    trans_dot: torch.Tensor | None  # (H, W)
    count: torch.Tensor  # (H, W) contributor count c
    coverage: torch.Tensor  # (H, W) b = sum_k w_k

    @property
    def alpha(self):
        return 1.0 - self.trans

    @property
    def alpha_dot(self):
        return None if self.trans_dot is None else -self.trans_dot


def _tile_lists(means2d, radius, depth, width, height, block=BLOCK):
    """Build per-tile, depth-sorted Gaussian index lists (padded with -1)."""
    device = means2d.device
    ntx = (width + block - 1) // block
    nty = (height + block - 1) // block
    ntiles = ntx * nty
    r = radius
    x, y = means2d[:, 0], means2d[:, 1]
    # getRect() from 3DGS (integer truncation toward zero, then clamp)
    x0 = torch.clamp(torch.trunc((x - r) / block), 0, ntx).long()
    y0 = torch.clamp(torch.trunc((y - r) / block), 0, nty).long()
    x1 = torch.clamp(torch.trunc((x + r + block - 1) / block), 0, ntx).long()
    y1 = torch.clamp(torch.trunc((y + r + block - 1) / block), 0, nty).long()
    wx = (x1 - x0).clamp(min=0)
    wy = (y1 - y0).clamp(min=0)
    cnt = wx * wy
    cnt = torch.where(radius > 0, cnt, torch.zeros_like(cnt))
    total = int(cnt.sum().item())
    if total == 0:
        return torch.full((ntiles, 1), -1, dtype=torch.long, device=device), torch.zeros(ntiles, dtype=torch.long, device=device), ntx, nty
    gid = torch.repeat_interleave(torch.arange(len(cnt), device=device), cnt)
    start = torch.cumsum(cnt, 0) - cnt
    k = torch.arange(total, device=device) - start[gid]
    tx = x0[gid] + k % wx[gid]
    ty = y0[gid] + k // wx[gid]
    tile = ty * ntx + tx
    # sort by (tile, depth), stable in Gaussian index like the 3DGS radix sort
    order = torch.argsort(depth[gid], stable=True)
    gid, tile = gid[order], tile[order]
    order = torch.argsort(tile, stable=True)
    gid, tile = gid[order], tile[order]
    lens = torch.bincount(tile, minlength=ntiles)
    lmax = int(lens.max().item())
    tstart = torch.cumsum(lens, 0) - lens
    pos = torch.arange(total, device=device) - tstart[tile]
    lists = torch.full((ntiles, lmax), -1, dtype=torch.long, device=device)
    lists[tile, pos] = gid
    return lists, lens, ntx, nty


def rasterize(
    proj: dict,
    tangents: dict | None,
    width: int,
    height: int,
    bg: torch.Tensor,
    chunk: int = 32,
    tile_batch: int = 4096,
):
    """Composite projected Gaussians (and their time-tangents) into an image.

    Args:
        proj: output of ``project_gaussians`` (primal values).
        tangents: same keys as ``proj`` holding d/dt of the continuous values, or
            None to render the primal image only.
        bg: (3,) background colour (assumed constant in time, so I_bg_dot = 0).
    """
    device = proj["means2d"].device
    means2d = proj["means2d"].detach()
    conic = proj["conic"].detach()
    color = proj["color"].detach()
    opac = proj["opacity"].detach()
    depth = proj["depth"].detach()
    radius = proj["radius"]
    with_dot = tangents is not None
    if with_dot:
        d_means2d = tangents["means2d"].detach()
        d_conic = tangents["conic"].detach()
        d_color = tangents["color"].detach()
        d_opac = tangents["opacity"].detach()

    lists, lens, ntx, nty = _tile_lists(means2d, radius, depth, width, height)
    ntiles = ntx * nty
    P = BLOCK * BLOCK
    ly, lx = torch.meshgrid(torch.arange(BLOCK, device=device), torch.arange(BLOCK, device=device), indexing="ij")
    lx, ly = lx.reshape(-1).float(), ly.reshape(-1).float()

    out_C = torch.zeros(ntiles, P, 3, device=device)
    out_dC = torch.zeros(ntiles, P, 3, device=device) if with_dot else None
    out_T = torch.ones(ntiles, P, device=device)
    out_R = torch.zeros(ntiles, P, device=device)  # T_dot / T
    out_n = torch.zeros(ntiles, P, device=device)
    out_done = torch.zeros(ntiles, P, dtype=torch.bool, device=device)

    tiles_all = torch.arange(ntiles, device=device)
    tile_px = (tiles_all % ntx).float() * BLOCK
    tile_py = (tiles_all // ntx).float() * BLOCK
    lmax = lists.shape[1]

    for s in range(0, lmax, chunk):
        active = (lens > s) & ~out_done.all(-1)
        act_idx = tiles_all[active]
        for b0 in range(0, len(act_idx), tile_batch):
            A = act_idx[b0 : b0 + tile_batch]
            idx = lists[A, s : s + chunk]  # (A, K)
            present = idx >= 0
            gi = idx.clamp(min=0)
            px = (tile_px[A][:, None] + lx[None, :]).unsqueeze(-1)  # (A, P, 1)
            py = (tile_py[A][:, None] + ly[None, :]).unsqueeze(-1)
            mx = means2d[gi, 0].unsqueeze(1)  # (A, 1, K)
            my = means2d[gi, 1].unsqueeze(1)
            dx = mx - px  # d = u_k - u  (3DGS convention)
            dy = my - py
            ca = conic[gi, 0].unsqueeze(1)
            cb = conic[gi, 1].unsqueeze(1)
            cc = conic[gi, 2].unsqueeze(1)
            power = -0.5 * (ca * dx * dx + cc * dy * dy) - cb * dx * dy
            g = torch.exp(power)
            sig = opac[gi].unsqueeze(1)
            raw = sig * g
            alpha = torch.clamp(raw, max=TAU_MAX)
            done_px = out_done[A].unsqueeze(-1)
            valid = present.unsqueeze(1) & (power <= 0) & (alpha >= ALPHA_MIN) & ~done_px
            tau = torch.where(valid, alpha, torch.zeros_like(alpha))

            T0 = out_T[A].unsqueeze(-1)
            Tinc = T0 * torch.cumprod(1.0 - tau, -1)
            keep = Tinc >= T_MIN  # prefix mask: early termination excludes the culprit
            newly_done = (~keep & valid).any(-1)
            tau = torch.where(keep, tau, torch.zeros_like(tau))
            valid = valid & keep
            one_m = 1.0 - tau
            Tinc = T0 * torch.cumprod(one_m, -1)
            Texc = Tinc / one_m
            w = Texc * tau
            col = color[gi].unsqueeze(1)  # (A, 1, K, 3)
            out_C[A] += (w.unsqueeze(-1) * col).sum(-2)
            out_n[A] += valid.sum(-1).float()

            if with_dot:
                ddx = d_means2d[gi, 0].unsqueeze(1)
                ddy = d_means2d[gi, 1].unsqueeze(1)
                dca = d_conic[gi, 0].unsqueeze(1)
                dcb = d_conic[gi, 1].unsqueeze(1)
                dcc = d_conic[gi, 2].unsqueeze(1)
                p_dot = (
                    -0.5 * (dca * dx * dx + dcc * dy * dy)
                    - dcb * dx * dy
                    - (ddx * (ca * dx + cb * dy) + ddy * (cb * dx + cc * dy))
                )
                dsig = d_opac[gi].unsqueeze(1)
                tau_dot = dsig * g + sig * g * p_dot
                tau_dot = torch.where(valid & (raw <= TAU_MAX), tau_dot, torch.zeros_like(tau_dot))
                ratio = tau_dot / one_m
                R0 = out_R[A].unsqueeze(-1)
                Rinc = R0 - torch.cumsum(ratio, -1)
                Rexc = Rinc + ratio
                w_dot = Texc * (Rexc * tau + tau_dot)
                dcol = d_color[gi].unsqueeze(1)
                out_dC[A] += (w_dot.unsqueeze(-1) * col + w.unsqueeze(-1) * dcol).sum(-2)
                out_R[A] = Rinc[..., -1]

            out_T[A] = Tinc[..., -1]
            out_done[A] = out_done[A] | newly_done

    def to_img(x):
        # (ntiles, P, ...) -> (H, W, ...)
        rest = x.shape[2:]
        x = x.reshape(nty, ntx, BLOCK, BLOCK, *rest).permute(0, 2, 1, 3, *range(4, 4 + len(rest)))
        x = x.reshape(nty * BLOCK, ntx * BLOCK, *rest)
        return x[:height, :width]

    T = to_img(out_T)
    C = to_img(out_C)
    rgb = C + T.unsqueeze(-1) * bg.to(device)
    trans_dot = rgb_dot = None
    if with_dot:
        trans_dot = T * to_img(out_R)
        rgb_dot = to_img(out_dC) + trans_dot.unsqueeze(-1) * bg.to(device)
    return RasterOutput(
        rgb=rgb,
        rgb_dot=rgb_dot,
        trans=T,
        trans_dot=trans_dot,
        count=to_img(out_n),
        coverage=1.0 - T,
    )
