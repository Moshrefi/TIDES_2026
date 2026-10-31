"""Camera intrinsics and a time-continuous camera trajectory.

Convention (same as 3DGS / COLMAP / OpenCV): a pose is world-to-camera,
``x_cam = R @ x_world + t``, with x right, y down, z forward. Pixel centres sit
at integer coordinates.

The trajectory is written in plain torch ops so that it can be differentiated
with respect to time by forward-mode AD (Sec. 3.2: "we treat the camera
parameters as time-dependent renderer inputs ... estimated from neighbouring
camera timestamps"). Translation uses a cubic Hermite spline whose tangents are
central differences of neighbouring poses, so the camera velocity is continuous
across keyframes. Rotation uses SLERP between neighbouring keyframes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled(self, s: float) -> "Intrinsics":
        """Intrinsics for an image resized by factor ``s`` (integer-centre convention)."""
        return Intrinsics(
            fx=self.fx * s,
            fy=self.fy * s,
            cx=(self.cx + 0.5) * s - 0.5,
            cy=(self.cy + 0.5) * s - 0.5,
            width=int(round(self.width * s)),
            height=int(round(self.height * s)),
        )

    def as_dict(self) -> dict:
        return dict(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, width=self.width, height=self.height)


# ----------------------------------------------------------------------------
# Rotation helpers (quaternions are (w, x, y, z))
# ----------------------------------------------------------------------------

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (..., 4) -> rotation matrix (..., 3, 3)."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], -1).reshape(q.shape[:-1] + (3, 3))


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (..., 3, 3) -> unit quaternion (..., 4), numpy, w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4))
    for i, m in enumerate(flat):
        tr = np.trace(m)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        q = np.array([w, x, y, z])
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        out[i] = q
    return out.reshape(R.shape[:-2] + (4,))


def _slerp(q0: torch.Tensor, q1: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Differentiable SLERP. q0, q1: (..., 4); s: (...,)."""
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    s = s.unsqueeze(-1)
    theta = torch.acos(dot.clamp(max=1.0 - 1e-7))
    sin_t = torch.sin(theta)
    small = sin_t < 1e-4
    w0 = torch.where(small, 1 - s, torch.sin((1 - s) * theta) / sin_t.clamp(min=1e-12))
    w1 = torch.where(small, s, torch.sin(s * theta) / sin_t.clamp(min=1e-12))
    q = w0 * q0 + w1 * q1
    return q / q.norm(dim=-1, keepdim=True)


class PoseTrajectory:
    """Continuous camera trajectory from timestamped world-to-camera poses.

    Args:
        times: (N,) seconds, strictly increasing.
        R_wc: (N, 3, 3) world-to-camera rotations.
        t_wc: (N, 3) world-to-camera translations.
        interp: "hermite" (C1 translation, default) or "linear".
    """

    def __init__(self, times, R_wc, t_wc, interp: str = "hermite", device="cpu", dtype=torch.float32):
        times = np.asarray(times, dtype=np.float64)
        R_wc = np.asarray(R_wc, dtype=np.float64)
        t_wc = np.asarray(t_wc, dtype=np.float64)
        assert times.ndim == 1 and len(times) >= 1
        if len(times) > 1:
            assert np.all(np.diff(times) > 0), "pose timestamps must be strictly increasing"
        # store camera-to-world (centre + orientation): smoother to interpolate
        R_cw = np.transpose(R_wc, (0, 2, 1))
        centres = -np.einsum("nij,nj->ni", R_cw, t_wc)
        q_cw = rotmat_to_quat(R_cw)
        # make quaternion signs consistent along the trajectory
        for i in range(1, len(q_cw)):
            if np.dot(q_cw[i], q_cw[i - 1]) < 0:
                q_cw[i] = -q_cw[i]
        # central-difference tangents for Hermite interpolation of the centre
        if len(times) > 1:
            tang = np.gradient(centres, times, axis=0)
        else:
            tang = np.zeros_like(centres)
        self.t0 = float(times[0])
        self.interp = interp
        self.device = device
        self.dtype = dtype
        # times relative to t0 in float64 numpy, tensors in float32 for maths
        self._times_np = times - self.t0
        self.times = torch.tensor(self._times_np, dtype=torch.float64, device=device)  # for searching
        self.times32 = torch.tensor(self._times_np, dtype=dtype, device=device)  # for arithmetic
        self.centres = torch.tensor(centres, dtype=dtype, device=device)
        self.tangents = torch.tensor(tang, dtype=dtype, device=device)
        self.quats = torch.tensor(q_cw, dtype=dtype, device=device)

    @classmethod
    def static(cls, R_wc, t_wc, device="cpu"):
        return cls(np.array([0.0]), np.asarray(R_wc)[None], np.asarray(t_wc)[None], device=device)

    @property
    def time_range(self):
        return self.t0, self.t0 + float(self._times_np[-1])

    def pose(self, t: torch.Tensor):
        """World-to-camera pose at times ``t`` (any shape, seconds, absolute).

        ``t`` may carry forward-mode tangents; the result is differentiable in t.
        Returns R_wc (..., 3, 3), t_wc (..., 3), camera centre (..., 3).
        """
        n = len(self._times_np)
        tr = t - self.t0
        if n == 1:
            shape = tuple(tr.shape)
            q = self.quats[0].expand(shape + (4,))
            c = self.centres[0].expand(shape + (3,)) + 0.0 * tr.unsqueeze(-1)
        else:
            with torch.no_grad():
                key = tr.detach().to(torch.float64)
                i = torch.searchsorted(self.times, key.reshape(-1), right=True).reshape(key.shape) - 1
                i = i.clamp(0, n - 2)
            # float32 copies: mixing float64 constants into a forward-mode graph
            # promotes the tangents to float64
            ta = self.times32[i]
            tb = self.times32[i + 1]
            h = tb - ta
            s = ((tr - ta) / h).clamp(0.0, 1.0)  # clamp: hold the end poses outside the range
            q = _slerp(self.quats[i], self.quats[i + 1], s)
            p0, p1 = self.centres[i], self.centres[i + 1]
            if self.interp == "linear":
                c = p0 + s.unsqueeze(-1) * (p1 - p0)
            else:
                m0 = self.tangents[i] * h.unsqueeze(-1)
                m1 = self.tangents[i + 1] * h.unsqueeze(-1)
                s1 = s.unsqueeze(-1)
                s2, s3 = s1 * s1, s1 * s1 * s1
                c = (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s1) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1
        R_cw = quat_to_rotmat(q)
        R_wc = R_cw.transpose(-1, -2)
        t_wc = -(R_wc @ c.unsqueeze(-1)).squeeze(-1)
        return R_wc, t_wc, c
