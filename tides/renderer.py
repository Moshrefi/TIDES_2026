"""Time-tangent renderer: (L, L_dot, alpha, alpha_dot, b, c) at any instant (Sec. 3.2).

Pipeline for one query time t (seconds from sequence start):

1. scene time  s = time_map(t)
2. Theta(s) from the deformation model, camera pose from the trajectory
3. projection to screen space (``projection.py``)
4. time-tangents of everything in 2-3 by a Jacobian-vector product along t
5. compositing of primal + tangent (``rasterizer.py``)
6. luminance Y = w^T I, L = log(Y + eps), L_dot = Y_dot / (Y + eps)   (Eq. 5, 15)

Tangent modes:
  * ``forward`` - torch.func.jvp (forward-mode AD), the paper's method.
  * ``reverse`` - exact fallback for networks whose ops lack forward-mode
    formulas: each Gaussian gets its own copy of t, and one reverse pass per
    continuous output channel (10 in total) gives d(out_k)/d(t_k) exactly.
  * ``fd_params`` - central finite differences of the per-Gaussian parameters
    (debug only; *not* the Table 4 finite-difference ablation, which lives in
    the simulator and differences rendered L).
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch

from .camera import Intrinsics, PoseTrajectory
from .projection import CONTINUOUS_KEYS, project_gaussians
from .rasterizer import rasterize

REC709 = (0.2126, 0.7152, 0.0722)


@dataclass
class Frame:
    t: float  # seconds
    L: torch.Tensor  # (H, W)
    L_dot: torch.Tensor | None
    Y: torch.Tensor
    rgb: torch.Tensor  # (H, W, 3)
    alpha: torch.Tensor
    alpha_dot: torch.Tensor | None
    count: torch.Tensor  # c(u,t)
    coverage: torch.Tensor  # b(u,t)


class TangentRenderer:
    def __init__(
        self,
        scene,
        trajectory: PoseTrajectory,
        K: Intrinsics,
        bg=(0.0, 0.0, 0.0),
        time_map=None,
        mode: str = "forward",
        eps: float = 1e-3,
        lum_weights=REC709,
        chunk: int = 32,
        fd_h: float = 1e-4,
        device=None,
    ):
        self.scene = scene
        self.traj = trajectory
        self.K = K
        self.device = device or trajectory.device
        self.bg = torch.tensor(bg, dtype=torch.float32, device=self.device)
        self.time_map = time_map if time_map is not None else (lambda t: t)
        self.mode = mode
        self.eps = eps
        self.lum = torch.tensor(lum_weights, dtype=torch.float32, device=self.device)
        self.chunk = chunk
        self.fd_h = fd_h
        self.num_queries = 0
        self._warned_no_time_grad = False

    # -- per-Gaussian screen-space primitives as a function of time ----------
    def _project(self, t_cam: torch.Tensor, t_scene_vec: torch.Tensor, shared: bool = False):
        p = self.scene.params(t_scene_vec)
        R, tv, c = self.traj.pose(t_cam)
        if shared:
            R, tv, c = R[0], tv[0], c[0]
        return project_gaussians(
            p["means"], p["scales_raw"], p["quats_raw"], p["opacity_raw"], p["shs"],
            self.scene.sh_degree, R, tv, c, self.K,
        )

    def _project_scalar(self, t: torch.Tensor):
        # t has shape (1,): 0-dim duals hit a dtype-promotion quirk in forward AD
        n = self.scene.num_gaussians
        return self._project(t, self.time_map(t).expand(n), shared=True)

    def primitives(self, t: float, with_tangent: bool = True):
        tt = torch.tensor([float(t)], dtype=torch.float32, device=self.device)
        if not with_tangent:
            with torch.no_grad():
                return self._project_scalar(tt), None
        if self.mode == "forward":
            def f(x):
                out = self._project_scalar(x)
                return tuple(out[k] for k in CONTINUOUS_KEYS), (out["radius"], out["valid"])

            with torch.no_grad():
                (prim, aux), (tang, _) = _jvp_with_aux(f, tt)
            proj = dict(zip(CONTINUOUS_KEYS, prim))
            proj["radius"], proj["valid"] = aux
            return proj, dict(zip(CONTINUOUS_KEYS, tang))
        if self.mode == "reverse":
            return self._reverse_tangents(tt)
        if self.mode == "fd_params":
            with torch.no_grad():
                proj = self._project_scalar(tt)
                h = self.fd_h
                a = self._project_scalar(tt + h)
                b = self._project_scalar(tt - h)
            return proj, {k: (a[k] - b[k]) / (2 * h) for k in CONTINUOUS_KEYS}
        raise ValueError(f"unknown tangent mode {self.mode}")

    def _reverse_tangents(self, tt):
        n = self.scene.num_gaussians
        tv = tt.detach().reshape(()).expand(n).clone().requires_grad_(True)
        with torch.enable_grad():
            out = self._project(tv, self.time_map(tv))
            # Frozen opacity / degree-zero SH colour can be genuine constants.
            # allow_unused handles an unused input, not an output without grad_fn.
            if not any(out[k].requires_grad for k in CONTINUOUS_KEYS) and not self._warned_no_time_grad:
                warnings.warn(
                    "All reverse-mode projection outputs are detached from autograd. "
                    "Time derivatives will be zero; verify that the scene/camera are static "
                    "and no time-dependent computation was accidentally detached.",
                    RuntimeWarning, stacklevel=2,
                )
                self._warned_no_time_grad = True
            tang = {}
            for k in CONTINUOUS_KEYS:
                y = out[k]
                if not y.requires_grad:
                    tang[k] = torch.zeros_like(y)
                    continue
                y2 = y.reshape(n, -1)
                cols = []
                for j in range(y2.shape[1]):
                    (g,) = torch.autograd.grad(y2[:, j].sum(), tv, retain_graph=True, allow_unused=True)
                    cols.append(torch.zeros_like(tv) if g is None else g)
                tang[k] = torch.stack(cols, -1).reshape(y.shape)
        proj = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}
        return proj, tang

    # -- full render --------------------------------------------------------
    def render(self, t: float, with_tangent: bool = True) -> Frame:
        proj, tang = self.primitives(t, with_tangent)
        with torch.no_grad():
            r = rasterize(proj, tang, self.K.width, self.K.height, self.bg, chunk=self.chunk)
            Y = torch.clamp((r.rgb * self.lum).sum(-1), min=0.0)
            L = torch.log(Y + self.eps)
            L_dot = None
            if with_tangent:
                Y_dot = (r.rgb_dot * self.lum).sum(-1)
                L_dot = Y_dot / (Y + self.eps)
        self.num_queries += 1
        return Frame(t=float(t), L=L, L_dot=L_dot, Y=Y, rgb=r.rgb, alpha=r.alpha,
                     alpha_dot=r.alpha_dot, count=r.count, coverage=r.coverage)


def _jvp_with_aux(f, t):
    """torch.func.jvp for a function returning (outputs_tuple, aux)."""
    def g(x):
        out, aux = f(x)
        return out, aux

    prim, tang, aux = torch.func.jvp(g, (t,), (torch.ones_like(t),), has_aux=True)
    return (prim, aux), (tang, None)
