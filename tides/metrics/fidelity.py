"""IG-NLL and motion-scaled spatiotemporal Chamfer distance (Sec. 4.3, Table 1)."""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Inter-spike intervals
# ----------------------------------------------------------------------------

def per_pixel_isi(ev, width: int, by_polarity: bool = False, unit: float = 1e-3):
    """Intervals between consecutive events at the same pixel, in ``unit`` seconds."""
    key = ev["y"].astype(np.int64) * width + ev["x"].astype(np.int64)
    if by_polarity:
        key = key * 2 + (ev["p"] > 0)
    order = np.lexsort((ev["t"], key))
    k, t = key[order], ev["t"][order]
    same = k[1:] == k[:-1]
    isi = (t[1:] - t[:-1])[same] / unit
    return isi


def fit_inverse_gaussian(x: np.ndarray):
    """MLE of IG(mu, lam): mu = mean(x), 1/lam = mean(1/x - 1/mu)."""
    x = x[x > 0]
    mu = x.mean()
    lam = 1.0 / max(np.mean(1.0 / x - 1.0 / mu), 1e-12)
    return mu, lam


def ig_logpdf(x, mu, lam):
    x = np.maximum(x, 1e-12)
    return 0.5 * np.log(lam / (2 * np.pi * x ** 3)) - lam * (x - mu) ** 2 / (2 * mu * mu * x)


def ig_nll(real, sim, width: int, unit: float = 1e-3, min_isi: float = 0.0, variant: str = "excess",
           by_polarity: bool = False):
    """Negative log-likelihood of the real ISIs under an IG fitted to the simulated ISIs.

    variant:
      "excess" (default) - NLL(real | IG_sim) - NLL(real | IG_real) >= 0, i.e. the
          extra cost of using the simulator's timing model. This is our best guess
          at the paper's normalisation (its values are ~1e-3 to 1e-1).
      "raw" - mean NLL(real | IG_sim) in the given time unit.
    """
    isi_r = per_pixel_isi(real, width, by_polarity, unit)
    isi_s = per_pixel_isi(sim, width, by_polarity, unit)
    isi_r = isi_r[isi_r > min_isi]
    isi_s = isi_s[isi_s > min_isi]
    if len(isi_r) < 2 or len(isi_s) < 2:
        return float("nan")
    mu_s, lam_s = fit_inverse_gaussian(isi_s)
    nll = -np.mean(ig_logpdf(isi_r, mu_s, lam_s))
    if variant == "raw":
        return float(nll)
    mu_r, lam_r = fit_inverse_gaussian(isi_r)
    return float(nll + np.mean(ig_logpdf(isi_r, mu_r, lam_r)))


# ----------------------------------------------------------------------------
# Motion-scaled Chamfer
# ----------------------------------------------------------------------------

def _voxel_unique(pts: np.ndarray, voxel: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def chamfer_st(real, sim, v: float, shape, window: float = 0.05, voxel=(1.0, 1.0, 1.0),
               normalize: str = "diag", max_points: int = 200_000, seed: int = 0):
    """Bidirectional nearest-neighbour distance between (x, y, v t) point sets.

    Events are polarity-matched (ON vs ON, OFF vs OFF) and compared within fixed
    temporal windows; each window's point sets are de-duplicated on a voxel grid
    (voxel given in embedded units: px, px, px-equivalent time). Distances are
    divided by the image diagonal ("diag"), width ("width") or not ("none").
    Returns the mean over windows and polarities.
    """
    from scipy.spatial import cKDTree

    H, W = shape
    scale = {"diag": float(np.hypot(H, W)), "width": float(W), "none": 1.0}[normalize]
    voxel = np.asarray(voxel, dtype=np.float64)
    t0 = max(real["t"].min(), sim["t"].min()) if len(real["t"]) and len(sim["t"]) else 0.0
    t1 = min(real["t"].max(), sim["t"].max()) if len(real["t"]) and len(sim["t"]) else 0.0
    rng = np.random.default_rng(seed)
    vals = []
    edges = np.arange(t0, t1, window)
    for a in edges:
        b = a + window
        for pol in (1, -1):
            sets = []
            for ev in (real, sim):
                i0, i1 = np.searchsorted(ev["t"], [a, b])
                sel = ev["p"][i0:i1] == pol
                p = np.stack([ev["x"][i0:i1][sel], ev["y"][i0:i1][sel], v * (ev["t"][i0:i1][sel] - a)], -1).astype(np.float64)
                p = _voxel_unique(p, voxel)
                if len(p) > max_points:
                    p = p[rng.choice(len(p), max_points, replace=False)]
                sets.append(p)
            A, B = sets
            if len(A) == 0 or len(B) == 0:
                continue
            dab, _ = cKDTree(B).query(A, k=1)
            dba, _ = cKDTree(A).query(B, k=1)
            vals.append(0.5 * (dab.mean() + dba.mean()) / scale)
    return float(np.mean(vals)) if vals else float("nan")
