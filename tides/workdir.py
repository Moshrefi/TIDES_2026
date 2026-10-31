"""A prepared sequence on disk ("work dir") and helpers to build a renderer from it.

Layout written by ``scripts/prepare_scene.py``::

    <work>/scene/images/000000.png ...      4DGS training images (undistorted, centred principal point)
    <work>/scene/sparse/0/{cameras,images,points3D}.txt
    <work>/tides_meta.json                  timing, cameras, poses, scale
    <work>/4dgs/                            4DGS output (after training)
    <work>/renders/                         frames for the baselines
    <work>/events/<method>.h5               simulated events
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .camera import Intrinsics, PoseTrajectory


def load_meta(work: str) -> dict:
    with open(os.path.join(work, "tides_meta.json")) as f:
        return json.load(f)


def render_trajectory(meta: dict, device="cpu") -> PoseTrajectory:
    """Trajectory of the camera we render into (the event camera)."""
    times = np.asarray(meta["poses"]["times_s"])
    R = np.asarray(meta["poses"]["R_wc"])
    t = np.asarray(meta["poses"]["t_wc"])
    T = np.asarray(meta["T_render_rgb"])
    R_ev = np.einsum("ij,njk->nik", T[:3, :3], R)
    t_ev = np.einsum("ij,nj->ni", T[:3, :3], t) + T[:3, 3]
    return PoseTrajectory(times, R_ev, t_ev, interp=meta.get("pose_interp", "hermite"), device=device)


def render_intrinsics(meta: dict, scale: float = 1.0) -> Intrinsics:
    K = Intrinsics(**meta["K_render"])
    return K.scaled(scale) if scale != 1.0 else K


def build_renderer(work: str, fourdgs_repo: str, model_path: str | None = None, config: str | None = None,
                   iteration: int = -1, mode: str = "forward", device="cuda", bg=None, eps: float = 1e-3,
                   chunk: int = 32):
    from .renderer import TangentRenderer
    from .scene.fourdgs import FourDGSScene, TimeMap

    meta = load_meta(work)
    tm = TimeMap(meta["frame_times_s"], meta["scene_times"], device=device)
    scene = FourDGSScene(model_path or os.path.join(work, "4dgs"), fourdgs_repo, config=config, iteration=iteration,
                         time_map=tm)
    traj = render_trajectory(meta, device=device)
    K = render_intrinsics(meta)
    if bg is None:
        bg = meta.get("background", (0.0, 0.0, 0.0))
    return TangentRenderer(scene, traj, K, bg=bg, time_map=tm, mode=mode, eps=eps, chunk=chunk, device=device), meta
