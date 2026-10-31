import numpy as np

from tides.baselines import ESIM
from tides.metrics import chamfer_st, fano_error, ig_nll, isi_spike_ratio, same_ts_fraction
from tides.metrics.fidelity import fit_inverse_gaussian


def poisson_events(n=20000, W=32, H=24, T=1.0, seed=0, quantum=None):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, T, n))
    if quantum:
        t = np.floor(t / quantum) * quantum
    return dict(t=t, x=rng.integers(0, W, n).astype(np.int32), y=rng.integers(0, H, n).astype(np.int32),
                p=np.where(rng.random(n) > 0.5, 1, -1).astype(np.int8))


def test_ig_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    mu, lam = 2.0, 5.0
    x = rng.wald(mu, lam, 200000)
    m, l = fit_inverse_gaussian(x)
    assert abs(m - mu) < 0.02 and abs(l - lam) / lam < 0.02


def test_identical_streams_score_perfectly():
    a = poisson_events()
    assert abs(ig_nll(a, a, 32)) < 1e-9
    assert chamfer_st(a, a, v=100.0, shape=(24, 32)) == 0.0
    assert fano_error(a, a, (24, 32)) < 1e-9


def test_batching_metrics_detect_quantised_timestamps():
    smooth = poisson_events(seed=1)
    batched = poisson_events(seed=1, quantum=1 / 300)  # all events on a 300 Hz render clock
    assert same_ts_fraction(batched) > 0.9
    assert same_ts_fraction(smooth) < 0.1
    assert isi_spike_ratio(batched, 32) > isi_spike_ratio(smooth, 32) + 0.5


def test_chamfer_grows_with_time_offset():
    a = poisson_events(seed=2)
    b = dict(a); b["t"] = a["t"] + 0.002
    c = dict(a); c["t"] = a["t"] + 0.01
    assert chamfer_st(a, b, 1000.0, (24, 32)) < chamfer_st(a, c, 1000.0, (24, 32))


def test_esim_fallback_batches_on_frame_times():
    """The frame-driven baseline places crossings on lines between frames: a fast
    ramp produces many events, and on a coarse frame grid timestamps cluster."""
    sim = ESIM((1, 1), c_pos=0.1, c_neg=0.1, device="cpu")
    for i, t in enumerate(np.linspace(0, 1, 11)):
        sim.feed(np.array([[0.1 + 0.08 * i]]), t)
    ev = sim.result()
    assert not sim.official
    assert len(ev["t"]) > 5 and (ev["p"] > 0).all()
