"""Per-Gaussian screen-space projection (EWA splatting), written in torch.

This mirrors the preprocessing of the reference 3DGS CUDA rasterizer
(``preprocessCUDA`` / ``computeCov2D`` / ``computeColorFromSH``) so that the
primal render matches the renderer the 4DGS scene was trained with. Because it
is plain torch, ``torch.func.jvp`` through it yields the per-Gaussian
time-tangents (u_k, du_k, Q_k, dQ_k, c_k, dc_k, sigma_k, dsigma_k) of Sec. 3.2.

Discrete decisions (culling, radius, tile rectangle) are taken from the primal
values only and carry no tangent.
"""
from __future__ import annotations

import torch

from .camera import Intrinsics, quat_to_rotmat

# Spherical-harmonic constants, identical to 3DGS (auxiliary.h / sh_utils.py)
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]

NEAR_PLANE = 0.2  # 3DGS in_frustum(): p_view.z <= 0.2 is culled
LOWPASS = 0.3  # 3DGS adds 0.3 px^2 to the 2D covariance diagonal


def eval_sh(deg: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """sh: (N, K, 3) with K >= (deg+1)^2, dirs: (N, 3) unit. Returns (N, 3)."""
    result = SH_C0 * sh[:, 0]
    if deg > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + SH_C2[0] * xy * sh[:, 4]
                + SH_C2[1] * yz * sh[:, 5]
                + SH_C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
                + SH_C2[3] * xz * sh[:, 7]
                + SH_C2[4] * (xx - yy) * sh[:, 8]
            )
            if deg > 2:
                result = (
                    result
                    + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9]
                    + SH_C3[1] * xy * z * sh[:, 10]
                    + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
                    + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
                    + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
                    + SH_C3[5] * z * (xx - yy) * sh[:, 14]
                    + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15]
                )
    return result


def project_gaussians(
    means: torch.Tensor,
    scales_raw: torch.Tensor,
    quats_raw: torch.Tensor,
    opacity_raw: torch.Tensor,
    shs: torch.Tensor,
    sh_degree: int,
    R_wc: torch.Tensor,
    t_wc: torch.Tensor,
    campos: torch.Tensor,
    K: Intrinsics,
    scale_modifier: float = 1.0,
):
    """Project 3D Gaussians to screen space.

    All per-Gaussian inputs are raw (pre-activation) parameters, as produced by
    the 4DGS deformation network (Eq. 6). Activations follow Eq. 7-8:
    s = exp(s_raw), sigma = sigmoid(sigma_raw), q = normalize(q_raw).

    Camera inputs may be shared (3,3)/(3,) or per-Gaussian (N,3,3)/(N,3); the
    per-Gaussian form is used by the reverse-mode tangent fallback.

    Returns a dict of continuous outputs (differentiable in time):
        means2d (N,2), conic (N,3) = (a, b, c) of the 2x2 precision Q,
        color (N,3), opacity (N,), depth (N,)
    """
    N = means.shape[0]
    scales = torch.exp(scales_raw) * scale_modifier
    opacity = torch.sigmoid(opacity_raw).reshape(N)
    rot = quat_to_rotmat(quats_raw)  # normalises internally

    # 3D covariance: Sigma = R S^2 R^T
    M = rot * scales.unsqueeze(-2)  # R @ diag(s)
    cov3d = M @ M.transpose(-1, -2)

    # to camera frame
    if R_wc.dim() == 2:
        p_view = means @ R_wc.T + t_wc
        W = R_wc.expand(N, 3, 3)
    else:
        p_view = (R_wc @ means.unsqueeze(-1)).squeeze(-1) + t_wc
        W = R_wc
    tx, ty, tz = p_view.unbind(-1)
    tz_safe = torch.where(tz > NEAR_PLANE, tz, torch.full_like(tz, 1.0))

    # computeCov2D (EWA, with the same off-screen clamp as 3DGS)
    tanfovx = K.width / (2.0 * K.fx)
    tanfovy = K.height / (2.0 * K.fy)
    limx, limy = 1.3 * tanfovx, 1.3 * tanfovy
    txtz = torch.clamp(tx / tz_safe, -limx, limx) * tz_safe
    tytz = torch.clamp(ty / tz_safe, -limy, limy) * tz_safe
    zero = torch.zeros_like(tz)
    J = torch.stack(
        [
            K.fx / tz_safe, zero, -(K.fx * txtz) / (tz_safe * tz_safe),
            zero, K.fy / tz_safe, -(K.fy * tytz) / (tz_safe * tz_safe),
        ],
        -1,
    ).reshape(N, 2, 3)
    T = J @ W
    cov2d = T @ cov3d @ T.transpose(-1, -2)
    a = cov2d[:, 0, 0] + LOWPASS
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + LOWPASS
    det = a * c - b * b
    det_safe = torch.where(det > 0, det, torch.ones_like(det))
    conic = torch.stack([c / det_safe, -b / det_safe, a / det_safe], -1)

    # projected centre (pixel centres at integer coordinates)
    u = K.fx * tx / tz_safe + K.cx
    v = K.fy * ty / tz_safe + K.cy
    means2d = torch.stack([u, v], -1)

    # view-dependent colour (computeColorFromSH); clamp at 0 like 3DGS
    if campos.dim() == 1:
        dirs = means - campos
    else:
        dirs = means - campos
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    rgb = eval_sh(sh_degree, shs, dirs) + 0.5
    color = torch.clamp(rgb, min=0.0)

    with torch.no_grad():
        valid = (tz > NEAR_PLANE) & (det > 0)
        mid = 0.5 * (a + c)
        lam1 = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
        radius = torch.ceil(3.0 * torch.sqrt(lam1))
        radius = torch.where(valid, radius, torch.zeros_like(radius))

    return dict(
        means2d=means2d,
        conic=conic,
        color=color,
        opacity=opacity,
        depth=tz,
        radius=radius.detach(),
        valid=valid.detach(),
    )


CONTINUOUS_KEYS = ("means2d", "conic", "color", "opacity", "depth")
