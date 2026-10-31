"""Event-stream container helpers.

Events are a dict of numpy arrays: t (float64 seconds), x, y (int), p (+1/-1),
sorted by t.
"""
from __future__ import annotations

import os

import numpy as np


def empty():
    return dict(t=np.zeros(0, np.float64), x=np.zeros(0, np.int32), y=np.zeros(0, np.int32), p=np.zeros(0, np.int8))


def concat(chunks):
    if not chunks:
        return empty()
    return {k: np.concatenate([c[k] for c in chunks]) for k in ("t", "x", "y", "p")}


def sort(ev):
    o = np.argsort(ev["t"], kind="stable")
    return {k: v[o] for k, v in ev.items()}


def slice_time(ev, t0, t1):
    i0, i1 = np.searchsorted(ev["t"], [t0, t1])
    return {k: v[i0:i1] for k, v in ev.items()}


def save(path, ev, meta: dict | None = None):
    """.npz or .h5 (t stored as int64 microseconds in h5, float64 seconds in npz)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".h5"):
        import h5py

        with h5py.File(path, "w") as f:
            g = f.create_group("events")
            g.create_dataset("t", data=np.round(ev["t"] * 1e6).astype(np.int64), compression="lzf")
            g.create_dataset("x", data=ev["x"].astype(np.uint16), compression="lzf")
            g.create_dataset("y", data=ev["y"].astype(np.uint16), compression="lzf")
            g.create_dataset("p", data=(ev["p"] > 0).astype(np.uint8), compression="lzf")
            for k, v in (meta or {}).items():
                f.attrs[k] = v
    else:
        np.savez_compressed(path, t=ev["t"], x=ev["x"].astype(np.int32), y=ev["y"].astype(np.int32),
                            p=ev["p"].astype(np.int8), **{f"meta_{k}": v for k, v in (meta or {}).items()})


def load(path):
    if path.endswith(".h5"):
        import h5py

        with h5py.File(path, "r") as f:
            g = f["events"]
            return dict(t=g["t"][:].astype(np.float64) * 1e-6, x=g["x"][:].astype(np.int32),
                        y=g["y"][:].astype(np.int32), p=np.where(g["p"][:] > 0, 1, -1).astype(np.int8))
    d = np.load(path)
    return dict(t=d["t"], x=d["x"], y=d["y"], p=d["p"])


def event_frame(ev, shape, t0=None, t1=None):
    """Red/blue polarity image (H, W, 3) uint8 for qualitative figures (Fig. 3)."""
    if t0 is not None:
        ev = slice_time(ev, t0, t1)
    H, W = shape
    img = np.full((H, W, 3), 255, np.uint8)
    pos = ev["p"] > 0
    img[ev["y"][pos], ev["x"][pos]] = (0, 0, 255)
    img[ev["y"][~pos], ev["x"][~pos]] = (255, 0, 0)
    return img
