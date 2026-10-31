"""Batching / burst diagnostics (Sec. 4.3, Table 2): Same-ts, ISI spike ratio, Fano error."""
from __future__ import annotations

import numpy as np

from .fidelity import per_pixel_isi


def same_ts_fraction(ev, window: float = 0.05, resolution: float = 1e-6):
    """Mean over windows of the fraction of events whose timestamp (at sensor
    resolution, 1 us by default) is shared with at least one other event."""
    if len(ev["t"]) == 0:
        return float("nan")
    q = np.round(ev["t"] / resolution).astype(np.int64)
    w = np.floor((ev["t"] - ev["t"][0]) / window).astype(np.int64)
    vals = []
    for wi in np.unique(w):
        qq = q[w == wi]
        if len(qq) < 2:
            continue
        _, inv, cnt = np.unique(qq, return_inverse=True, return_counts=True)
        vals.append(np.mean(cnt[inv] > 1))
    return float(np.mean(vals)) if vals else 0.0


def isi_spike_ratio(ev, width: int, zero_thresh: float = 5e-6, bin_size: float = 1e-6,
                    max_isi: float = 0.05, peak_factor: float = 5.0, half_window: int = 25):
    """Fraction of per-pixel ISIs that are near zero or fall in 'clock-quantum' peaks.

    A histogram of ISIs (1 us bins up to ``max_isi``) is compared to its local
    running median; bins more than ``peak_factor`` times the local median are
    treated as quantisation spikes.
    """
    isi = per_pixel_isi(ev, width, unit=1.0)
    if len(isi) == 0:
        return float("nan")
    near_zero = isi < zero_thresh
    rest = isi[(~near_zero) & (isi < max_isi)]
    spike_mass = 0
    if len(rest):
        nb = int(np.ceil(max_isi / bin_size))
        hist = np.bincount(np.minimum((rest / bin_size).astype(np.int64), nb - 1), minlength=nb).astype(np.float64)
        from scipy.ndimage import median_filter

        local = median_filter(hist, size=2 * half_window + 1, mode="nearest")
        spikes = hist > peak_factor * np.maximum(local, 1.0)
        spike_mass = hist[spikes].sum()
    return float((near_zero.sum() + spike_mass) / len(isi))


def _fano_per_tile(ev, shape, tile: int, t0: float, t1: float, dt: float):
    H, W = shape
    nty, ntx = (H + tile - 1) // tile, (W + tile - 1) // tile
    nb = max(1, int(np.floor((t1 - t0) / dt)))
    sel = (ev["t"] >= t0) & (ev["t"] < t0 + nb * dt)
    ti = ((ev["t"][sel] - t0) / dt).astype(np.int64)
    m = (ev["y"][sel] // tile) * ntx + ev["x"][sel] // tile
    counts = np.zeros((nty * ntx, nb))
    np.add.at(counts, (m, ti), 1)
    mean = counts.mean(1)
    var = counts.var(1)
    return mean, var


def fano_error(real, sim, shape, tile: int = 8, dt: float = 0.01, eps: float = 1e-3, min_mean: float = 0.1):
    """E_Fano = mean_b |log(F_b^pred / F_b^gt)| (Eq. 20).

    Bins b are spatial tiles (tile x tile px); within each tile the Fano factor
    Var(N)/E[N] is taken over the event counts N of consecutive ``dt`` slices.
    Tiles where either stream has mean count < ``min_mean`` are skipped.
    """
    t0 = max(real["t"].min(), sim["t"].min())
    t1 = min(real["t"].max(), sim["t"].max())
    mr, vr = _fano_per_tile(real, shape, tile, t0, t1, dt)
    ms, vs = _fano_per_tile(sim, shape, tile, t0, t1, dt)
    ok = (mr >= min_mean) & (ms >= min_mean)
    if not ok.any():
        return float("nan")
    Fr = vr[ok] / mr[ok] + eps
    Fs = vs[ok] / ms[ok] + eps
    return float(np.mean(np.abs(np.log(Fs / Fr))))
