"""Common dataset structures.

A ``Sequence`` describes one recording with paired RGB frames and real events:

* RGB frames + timestamps (used to fit the 4DGS scene)
* the RGB camera intrinsics/distortion
* the event camera intrinsics/distortion and the rigid transform from the RGB
  camera to the event camera (identity for datasets whose events are already
  aligned to the frames)
* a loader for the real events in a time range, already mapped into the
  undistorted event camera that TIDES and the baselines render into

All timestamps are float64 seconds relative to ``t_origin`` (the first frame).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np

from ..camera import Intrinsics


@dataclass
class CameraCalib:
    K: Intrinsics
    dist: np.ndarray = field(default_factory=lambda: np.zeros(0))  # OpenCV radtan (k1,k2,p1,p2[,k3]) or empty
    model: str = "radtan"  # "radtan" | "equidistant"


@dataclass
class Sequence:
    dataset: str
    name: str
    frame_paths: list
    frame_times: np.ndarray  # seconds, relative to t_origin
    t_origin: float  # absolute seconds of the first frame
    rgb: CameraCalib | None
    event: CameraCalib | None
    T_ev_rgb: np.ndarray = field(default_factory=lambda: np.eye(4))  # x_ev = T @ x_rgb
    gt_poses: tuple | None = None  # (times_s_rel, R_wc (N,3,3), t_wc (N,3)) of the RGB camera
    event_loader: object = None  # callable(t0_rel, t1_rel) -> event dict in event-camera pixels (undistorted)
    event_shape: tuple | None = None  # (H, W)
    notes: str = ""
    calibration_path: str | None = None

    def load_events(self, t0: float, t1: float) -> dict:
        return self.event_loader(t0, t1)


def to_seconds(ts: np.ndarray, hint: str = "auto") -> np.ndarray:
    """Convert integer/float timestamps to seconds (auto-detect ns / us / s)."""
    ts = np.asarray(ts, dtype=np.float64)
    if hint == "us":
        return ts * 1e-6
    if hint == "ns":
        return ts * 1e-9
    if hint == "s":
        return ts
    span = np.nanmax(ts) - np.nanmin(ts) if len(ts) > 1 else abs(ts[0])
    mag = np.nanmax(np.abs(ts))
    if mag > 1e17 or span > 1e11:
        return ts * 1e-9
    if mag > 1e11 or span > 1e5:
        return ts * 1e-6
    return ts


def sorted_images(folder: str, exts=(".png", ".jpg", ".jpeg")):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, "*" + e))
    return sorted(files)


def undistort_events(ev: dict, calib: CameraCalib, K_out: Intrinsics) -> dict:
    """Map raw event pixels through the inverse distortion into ``K_out`` pixels."""
    if calib is None or calib.dist is None or len(calib.dist) == 0 or not np.any(calib.dist):
        x, y = ev["x"].astype(np.float64), ev["y"].astype(np.float64)
        if K_out is not None and calib is not None and (K_out.fx != calib.K.fx or K_out.cx != calib.K.cx):
            x = (x - calib.K.cx) / calib.K.fx * K_out.fx + K_out.cx
            y = (y - calib.K.cy) / calib.K.fy * K_out.fy + K_out.cy
    else:
        import cv2

        Kin = np.array([[calib.K.fx, 0, calib.K.cx], [0, calib.K.fy, calib.K.cy], [0, 0, 1]])
        Ko = np.array([[K_out.fx, 0, K_out.cx], [0, K_out.fy, K_out.cy], [0, 0, 1]])
        # undistort each unique pixel once
        key = ev["y"].astype(np.int64) * 100000 + ev["x"].astype(np.int64)
        uk, inv = np.unique(key, return_inverse=True)
        pts = np.stack([uk % 100000, uk // 100000], -1).astype(np.float64)[:, None, :]
        if calib.model == "equidistant":
            und = cv2.fisheye.undistortPoints(pts, Kin, calib.dist[:4], P=Ko)
        else:
            und = cv2.undistortPoints(pts, Kin, calib.dist, P=Ko)
        und = und[:, 0, :][inv]
        x, y = und[:, 0], und[:, 1]
    xi, yi = np.round(x).astype(np.int64), np.round(y).astype(np.int64)
    W, H = K_out.width, K_out.height
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    return dict(t=ev["t"][ok], x=xi[ok].astype(np.int32), y=yi[ok].astype(np.int32), p=ev["p"][ok])


def find_h5_events(f):
    """Locate x, y, t, p datasets in an HDF5 file with unknown layout."""
    names = {}

    def visit(name, obj):
        leaf = name.split("/")[-1]
        if hasattr(obj, "shape") and leaf in ("x", "y", "t", "p", "ts", "polarity", "timestamp"):
            names.setdefault({"ts": "t", "timestamp": "t", "polarity": "p"}.get(leaf, leaf), name)

    f.visititems(visit)
    missing = {"x", "y", "t", "p"} - set(names)
    if missing:
        raise KeyError(f"could not find event fields {missing} in {f.filename}")
    return names
