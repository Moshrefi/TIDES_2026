"""Correctness tests for the core TIDES pieces. Run: pytest -q tests/"""
import math

import numpy as np
import pytest
import torch

from tides.camera import Intrinsics, PoseTrajectory, quat_to_rotmat
from tides.comparator import Comparator, ComparatorConfig
from tides.projection import project_gaussians
from tides.rasterizer import ALPHA_MIN, T_MIN, TAU_MAX, rasterize
from tides.readout import ReadoutConfig, TileArbiter
from tides.renderer import TangentRenderer
from tides.scene import ToyScene
from tides.simulator import TidesConfig, TidesSimulator

torch.set_num_threads(4)
K = Intrinsics(fx=40.0, fy=40.0, cx=23.5, cy=17.5, width=48, height=36)


def moving_camera():
    times = np.linspace(0, 1, 6)
    R = np.stack([quat_to_rotmat(torch.tensor([1.0, 0.02 * i, 0.03 * i, 0.0])).numpy() for i in range(6)])
    t = np.stack([[0.1 * i, -0.05 * i, 0.0] for i in range(6)])
    return PoseTrajectory(times, R, t)


def make_renderer(mode="forward", seed=0, n=60, **kw):
    scene = ToyScene.random(n=n, seed=seed, scale_freq=0.7, opacity_freq=0.9, **kw)
    return TangentRenderer(scene, moving_camera(), K, bg=(0.2, 0.3, 0.4), mode=mode)


# ----------------------------------------------------------------------------
def brute_force_render(proj, W, H, bg):
    """Per-pixel reference compositing straight from the 3DGS rules (no tiles)."""
    m = proj["means2d"].numpy(); con = proj["conic"].numpy(); col = proj["color"].numpy()
    op = proj["opacity"].numpy(); dep = proj["depth"].numpy(); rad = proj["radius"].numpy()
    order = np.argsort(dep, kind="stable")
    img = np.zeros((H, W, 3)); Tm = np.ones((H, W))
    for y in range(H):
        for x in range(W):
            T = 1.0; C = np.zeros(3)
            tx, ty = x // 16, y // 16
            for g in order:
                r = rad[g]
                if r <= 0:
                    continue
                x0 = min(math.ceil(W / 16), max(0, int((m[g, 0] - r) / 16)))
                x1 = min(math.ceil(W / 16), max(0, int((m[g, 0] + r + 15) / 16)))
                y0 = min(math.ceil(H / 16), max(0, int((m[g, 1] - r) / 16)))
                y1 = min(math.ceil(H / 16), max(0, int((m[g, 1] + r + 15) / 16)))
                if not (x0 <= tx < x1 and y0 <= ty < y1):
                    continue
                dx, dy = m[g, 0] - x, m[g, 1] - y
                power = -0.5 * (con[g, 0] * dx * dx + con[g, 2] * dy * dy) - con[g, 1] * dx * dy
                if power > 0:
                    continue
                a = min(TAU_MAX, op[g] * math.exp(power))
                if a < ALPHA_MIN:
                    continue
                if T * (1 - a) < T_MIN:
                    break
                C += col[g] * a * T
                T *= 1 - a
            img[y, x] = C + T * bg
            Tm[y, x] = T
    return img, Tm


def test_rasterizer_matches_bruteforce():
    r = make_renderer()
    proj, _ = r.primitives(0.3, with_tangent=False)
    out = rasterize(proj, None, K.width, K.height, r.bg)
    ref, Tm = brute_force_render(proj, K.width, K.height, r.bg.numpy())
    assert np.abs(out.rgb.numpy() - ref).max() < 1e-4
    assert np.abs(out.trans.numpy() - Tm).max() < 1e-4


@pytest.mark.parametrize("mode", ["forward", "reverse"])
def test_tangent_matches_finite_difference(mode):
    """d/dt of the rendered image vs central differences of the primal render."""
    r = make_renderer(mode)
    t, h = 0.37, 1e-3
    f = r.render(t)
    fp = r.render(t + h, with_tangent=False)
    fm = r.render(t - h, with_tangent=False)
    fd = (fp.Y - fm.Y) / (2 * h)
    an = (f.L_dot * (f.Y + r.eps))  # = Y_dot
    # exclude pixels where a discrete decision (alpha cut-off, tile list) flips inside +-h
    smooth = ((fp.Y - 2 * f.Y + fm.Y).abs() < 1e-4) & (f.count > 0)
    err = (fd - an)[smooth].abs()
    scale = an[smooth].abs().mean()
    assert smooth.float().mean() > 0.7
    assert float(err.median()) < 1e-2 * float(scale) + 1e-5
    assert float(torch.quantile(err, 0.95)) < 5e-2 * float(scale) + 1e-4


def test_forward_and_reverse_agree():
    a = make_renderer("forward").render(0.5)
    b = make_renderer("reverse").render(0.5)
    assert torch.allclose(a.L, b.L, atol=1e-6)
    assert torch.allclose(a.L_dot, b.L_dot, atol=1e-4, rtol=1e-3)
    assert torch.allclose(a.alpha_dot, b.alpha_dot, atol=1e-4, rtol=1e-3)


def test_alpha_dot_matches_fd():
    r = make_renderer()
    t, h = 0.53, 1e-3  # away from pose keyframes (rotation SLERP has a kink there)
    f = r.render(t)
    ap, am = r.render(t + h, False).alpha, r.render(t - h, False).alpha
    da = (ap - am) / (2 * h)
    smooth = (f.count > 0) & ((ap - 2 * f.alpha + am).abs() < 1e-4)
    assert torch.corrcoef(torch.stack([da[smooth], f.alpha_dot[smooth]]))[0, 1] > 0.99


# ----------------------------------------------------------------------------
def test_comparator_linear_exact_times():
    cfg = ComparatorConfig(c_pos=0.2, c_neg=0.2, n_max=100, model="linear")
    comp = Comparator(cfg, (1, 2))
    S0 = torch.zeros(1, 2)
    slope = torch.tensor([[1.0, -0.5]])
    comp.reset(S0)
    ev = comp.step(10.0, 1.0, S0, slope)
    t = ev["t"].numpy(); x = ev["x"].numpy(); p = ev["p"].numpy()
    t0 = np.sort(t[x == 0]); t1 = np.sort(t[x == 1])
    np.testing.assert_allclose(t0, 10.0 + 0.2 * np.arange(1, 6), atol=1e-6)
    np.testing.assert_allclose(t1, 10.0 + 0.4 * np.arange(1, 3), atol=1e-6)
    assert (p[x == 0] == 1).all() and (p[x == 1] == -1).all()


def test_comparator_quadratic_vs_dense():
    """Non-monotonic quadratic window: compare with brute-force dense sampling."""
    cfg = ComparatorConfig(c_pos=0.1, c_neg=0.1, n_max=50, model="quadratic")
    comp = Comparator(cfg, (1, 1))
    dt = 1.05
    S = lambda d: 1.0 * d - 1.2 * d * d  # rises to ~0.208 then falls to ~-0.27
    S0 = torch.tensor([[0.0]]); Sd = torch.tensor([[1.0]]); S1 = torch.tensor([[S(dt)]])
    comp.reset(S0)
    ev = comp.step(0.0, dt, S0, Sd, S1)
    # dense reference
    ds = np.linspace(0, dt, 200001)
    ref, out = 0.0, []
    for d in ds[1:]:
        v = S(d)
        if v - ref >= 0.1:
            out.append((d, 1)); ref += 0.1
        elif ref - v >= 0.1:
            out.append((d, -1)); ref -= 0.1
    assert len(out) == len(ev["t"])
    np.testing.assert_allclose(ev["t"].numpy(), [o[0] for o in out], atol=2e-5)
    assert list(ev["p"].numpy()) == [o[1] for o in out]


def test_comparator_nmax_and_catchup():
    cfg = ComparatorConfig(c_pos=0.1, c_neg=0.1, n_max=1, model="linear")
    comp = Comparator(cfg, (1, 1))
    S0 = torch.zeros(1, 1); comp.reset(S0)
    ev = comp.step(0.0, 1.0, S0, torch.tensor([[0.35]]))
    assert len(ev["t"]) == 1
    # next window starts at 0.35: two thresholds are pending -> emitted at window start
    cfg.n_max = 8
    ev2 = comp.step(1.0, 1.0, torch.tensor([[0.35]]), torch.zeros(1, 1))
    assert len(ev2["t"]) == 2 and np.allclose(ev2["t"].numpy(), 1.0)


# ----------------------------------------------------------------------------
def test_arbiter_capacity_and_causality():
    cfg = ReadoutConfig(enabled=True, tile=4, tick=1e-6, K0=2, beta_K=0.0, p0=0.0, eta=0.0, sigma0=0.0, gamma_sigma=0.0)
    arb = TileArbiter(cfg, (4, 4))
    arb.record(0.0, 1.0, torch.zeros(1, 1))
    n = 10
    ev = dict(t=np.full(n, 1e-3), x=np.zeros(n, np.int32), y=np.zeros(n, np.int32), p=np.ones(n, np.int8))
    out = arb(ev)
    assert len(out["t"]) == n
    assert (out["t"] >= 1e-3).all()
    ticks = np.round((out["t"] - 1e-3) / 1e-6).astype(int)
    assert np.bincount(ticks).max() <= 2  # at most K0 per tick
    assert ticks.max() == n // 2 - 1


def test_arbiter_drops():
    cfg = ReadoutConfig(enabled=True, tile=4, p0=0.5, eta=0.0)
    arb = TileArbiter(cfg, (4, 4))
    arb.record(0.0, 1.0, torch.zeros(1, 1))
    n = 20000
    ev = dict(t=np.sort(np.random.rand(n)), x=np.zeros(n, np.int32), y=np.zeros(n, np.int32), p=np.ones(n, np.int8))
    out = arb(ev)
    assert abs(len(out["t"]) / n - 0.5) < 0.02


# ----------------------------------------------------------------------------
@pytest.mark.parametrize("adaptive", [False, True])
def test_end_to_end_toy(adaptive):
    r = make_renderer()
    cfg = TidesConfig.from_dict({"adaptive": {"enabled": adaptive}, "readout": {"enabled": True, "tile": 16}})
    sim = TidesSimulator(r, cfg, frame_interval=0.05)
    ev = sim.run(0.0, 0.2, progress=False)
    assert len(ev["t"]) > 100
    assert np.all(np.diff(ev["t"]) >= 0)
    assert ev["x"].max() < K.width and ev["y"].max() < K.height
    # continuous timestamps: very few events share a timestamp
    same = 1 - len(np.unique(ev["t"])) / len(ev["t"])
    assert same < 0.05
    if adaptive:
        assert sim.stats["min_dt"] < 0.05 * 0.25 + 1e-12


def test_finite_difference_ablation_runs():
    r = make_renderer()
    cfg = TidesConfig.from_dict({"slope": "finite_difference", "readout": {"enabled": False}})
    ev = TidesSimulator(r, cfg, frame_interval=0.05).run(0.0, 0.1, progress=False)
    assert len(ev["t"]) > 0
