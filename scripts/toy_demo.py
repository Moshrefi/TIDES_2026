#!/usr/bin/env python
"""Self-contained sanity check on a synthetic scene (no datasets, no 4DGS, CPU ok).

A small analytic Gaussian scene with moving, overlapping splats is rendered by
the TIDES renderer. "Ground truth" events come from a brute-force simulation
with a very fine step (frame interval / 400). We then compare, at the same 10x
query budget, TIDES against an ESIM-style frame-differencing simulator.

  python scripts/toy_demo.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tides.baselines import ESIM  # noqa: E402
from tides.camera import Intrinsics, PoseTrajectory, quat_to_rotmat  # noqa: E402
from tides.metrics import chamfer_st, fano_error, ig_nll, isi_spike_ratio, same_ts_fraction  # noqa: E402
from tides.renderer import TangentRenderer  # noqa: E402
from tides.scene import ToyScene  # noqa: E402
from tides.simulator import TidesConfig, TidesSimulator  # noqa: E402


def main():
    torch.manual_seed(0)
    K = Intrinsics(fx=60.0, fy=60.0, cx=31.5, cy=23.5, width=64, height=48)
    times = np.linspace(0, 1, 5)
    R = np.stack([quat_to_rotmat(torch.tensor([1.0, 0.0, 0.05 * i, 0.0])).numpy() for i in range(5)])
    t = np.stack([[0.15 * i, 0.0, 0.0] for i in range(5)])
    traj = PoseTrajectory(times, R, t)
    scene = ToyScene.random(n=150, seed=3, speed=1.5, spread=1.6)
    r = TangentRenderer(scene, traj, K, bg=(1.0, 1.0, 1.0))
    frame_dt = 1 / 30
    T0, T1 = 0.1, 0.4

    def run(cfg_dict, fi):
        cfg = TidesConfig.from_dict(cfg_dict)
        sim = TidesSimulator(r, cfg, frame_interval=fi)
        ev = sim.run(T0, T1, progress=False)
        return ev, sim.stats

    print("ground truth (step = frame/400, ideal readout) ...")
    gt, _ = run({"readout": {"enabled": False}, "comparator": {"n_max": 64}, "adaptive": {"fixed_fraction": 1 / 400}}, frame_dt)
    print("TIDES 10x ...")
    tides, st = run({"readout": {"enabled": False}, "adaptive": {"fixed_fraction": 0.1}}, frame_dt)
    print("TIDES adaptive ...")
    tides_a, st_a = run({"readout": {"enabled": False}, "adaptive": {"enabled": True}}, frame_dt)
    print("ESIM-style 10x ...")
    esim = ESIM((K.height, K.width), device="cpu")
    for tt in np.arange(T0, T1 + 1e-9, frame_dt / 10):
        esim.feed(r.render(float(tt), with_tangent=False).Y.numpy(), float(tt))
    es = esim.result()

    v = 60.0
    rows = []
    for name, ev in [("ESIM-style (10x)", es), ("TIDES (10x)", tides), ("TIDES (adaptive)", tides_a)]:
        rows.append((name, len(ev["t"]), ig_nll(gt, ev, K.width), chamfer_st(gt, ev, v, (K.height, K.width), window=0.05),
                     same_ts_fraction(ev), isi_spike_ratio(ev, K.width), fano_error(gt, ev, (K.height, K.width), dt=0.01)))
    print(f"\nground truth: {len(gt['t'])} events; same-ts {same_ts_fraction(gt):.4f}")
    print(f"{'method':18s} {'events':>7s} {'IG-NLL':>9s} {'Chamfer':>9s} {'Same-ts':>8s} {'ISI-spk':>8s} {'Fano':>7s}")
    for r_ in rows:
        print(f"{r_[0]:18s} {r_[1]:7d} {r_[2]:9.5f} {r_[3]:9.5f} {r_[4]:8.4f} {r_[5]:8.4f} {r_[6]:7.3f}")
    print(f"\nTIDES adaptive used {st_a['render_queries']} queries vs {st['render_queries']} for 10x")


if __name__ == "__main__":
    main()
