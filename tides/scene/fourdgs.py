"""Wrapper around a trained 4DGS model (Wu et al., CVPR 2024; the paper's ref [37]).

The upstream code (https://github.com/hustvl/4DGaussians) is used unmodified to
load the checkpoint and run the deformation network; TIDES only calls the
deformation network and does its own projection and rasterization.
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from argparse import ArgumentParser

import numpy as np
import torch

from .base import DeformableScene


class TimeMap:
    """Piecewise-linear map from seconds (relative to sequence start) to scene time.

    4DGS's COLMAP reader assigns frame i the scene time i / N. ``frame_times``
    are the matching timestamps in seconds. Differentiable in t.
    """

    def __init__(self, frame_times_s, scene_times, device="cpu"):
        self.ft = torch.tensor(np.asarray(frame_times_s, np.float64), device=device)
        self.ft32 = self.ft.float()
        self.st32 = torch.tensor(np.asarray(scene_times, np.float64), device=device).float()
        if (self.ft.ndim != 1 or self.st32.shape != self.ft.shape or len(self.ft) < 2
                or not torch.isfinite(self.ft).all() or not torch.isfinite(self.st32).all()
                or not torch.all(self.ft[1:] > self.ft[:-1])
                or not torch.all(self.st32[1:] > self.st32[:-1])):
            raise ValueError("TimeMap requires matching finite, strictly increasing timestamps")

    @classmethod
    def from_json(cls, path, device="cpu"):
        with open(path) as f:
            d = json.load(f)
        return cls(d["frame_times_s"], d["scene_times"], device=device)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            key = t.detach().to(torch.float64).reshape(-1)
            i = (torch.searchsorted(self.ft, key, right=True) - 1).clamp(0, len(self.ft) - 2).reshape(t.shape)
        f0, f1 = self.ft32[i], self.ft32[i + 1]
        s0, s1 = self.st32[i], self.st32[i + 1]
        return s0 + (t - f0) * (s1 - s0) / (f1 - f0)


def _load_upstream(repo_path: str):
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from arguments import ModelHiddenParams  # noqa: E402
    from scene.gaussian_model import GaussianModel  # noqa: E402

    return ModelHiddenParams, GaussianModel


class FourDGSScene(DeformableScene):
    """A trained 4DGS scene.

    Args:
        model_path: the 4DGS output folder (contains ``point_cloud/iteration_*``).
        repo_path: path to a checkout of hustvl/4DGaussians.
        config: the same ``arguments/<...>.py`` config used for training (optional).
        iteration: checkpoint iteration (-1 = latest).
        time_map: maps seconds to the scene's time axis.
    """

    def __init__(self, model_path, repo_path, config=None, iteration=-1, sh_degree=3, time_map: TimeMap | None = None):
        ModelHiddenParams, GaussianModel = _load_upstream(repo_path)
        parser = ArgumentParser()
        mh = ModelHiddenParams(parser)
        hp = mh.extract(parser.parse_args([]))
        if config:
            cfg = runpy.run_path(config)
            for k, v in cfg.get("ModelHiddenParams", {}).items():
                setattr(hp, k, v)
        if iteration == -1:
            its = [int(d.split("_")[-1]) for d in os.listdir(os.path.join(model_path, "point_cloud")) if d.startswith("iteration_")]
            iteration = max(its)
        ckpt = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
        gm = GaussianModel(sh_degree, hp)
        gm.load_ply(os.path.join(ckpt, "point_cloud.ply"))
        gm.load_model(ckpt)
        gm._deformation.eval()
        for p in gm._deformation.parameters():
            p.requires_grad_(False)
        self.gm = gm
        self.sh_degree = gm.active_sh_degree
        self.num_gaussians = gm.get_xyz.shape[0]
        self.time_map = time_map
        # cache detached canonical parameters
        self._xyz = gm._xyz.detach()
        self._scaling = gm._scaling.detach()
        self._rotation = gm._rotation.detach()
        self._opacity = gm._opacity.detach()
        self._shs = gm.get_features.detach()

    def params(self, times):
        """times: (P,) scene time. Returns raw deformed parameters."""
        t = times.reshape(-1, 1)
        mu, s, q, o, sh = self.gm._deformation(self._xyz, self._scaling, self._rotation, self._opacity, self._shs, t)
        return dict(means=mu, scales_raw=s, quats_raw=q, opacity_raw=o.reshape(-1), shs=sh)
