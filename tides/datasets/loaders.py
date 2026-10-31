"""Loaders for the four datasets in Sec. 4.2: EDS, DSEC, HS-ERGB, BS-ERGB.

File layouts follow the public releases as we know them. Each loader prints
what it found; run ``python scripts/inspect_dataset.py`` on one sequence first
and adjust the paths below if your copy is organised differently.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from ..camera import Intrinsics
from .base import CameraCalib, Sequence, find_h5_events, sorted_images, to_seconds, undistort_events

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _read_txt_column(path, col=0, comments="#"):
    a = np.loadtxt(path, comments=comments, ndmin=2, delimiter=None)
    return a[:, col]


def _tum_to_w2c(arr):
    """TUM rows (t, tx, ty, tz, qx, qy, qz, qw), camera-to-world -> (R_wc, t_wc)."""
    from ..camera import quat_to_rotmat
    import torch

    q = torch.tensor(arr[:, [7, 4, 5, 6]], dtype=torch.float64)  # w, x, y, z
    R_cw = quat_to_rotmat(q).numpy()
    c = arr[:, 1:4]
    R_wc = np.transpose(R_cw, (0, 2, 1))
    t_wc = -np.einsum("nij,nj->ni", R_wc, c)
    return R_wc, t_wc


# ============================================================================
# EDS  (Hidalgo-Carrio et al., CVPR 2022)
# <root>/<seq>/images/*.png, images_timestamps.txt, events.h5, stamped_groundtruth.txt
# <root>/calib.yaml (kalibr: cam0 = RGB, cam1 = event camera)
# ============================================================================

def load_eds(root: str, seq: str, gt_pose_frame: str = "rgb") -> Sequence:
    import h5py

    d = os.path.join(root, seq)
    frames = sorted_images(os.path.join(d, "images"))
    ts_file = next(p for p in [os.path.join(d, "images_timestamps.txt"), os.path.join(d, "images", "timestamps.txt")] if os.path.exists(p))
    ft = to_seconds(_read_txt_column(ts_file, -1 if np.loadtxt(ts_file, ndmin=2).shape[1] > 1 else 0))
    assert len(ft) == len(frames), f"{len(ft)} timestamps vs {len(frames)} images"
    t_origin = float(ft[0])

    calib_path = next((p for p in [os.path.join(d, "calib.yaml"), os.path.join(root, "calib.yaml")] if os.path.exists(p)), None)
    rgb = ev_cal = None
    T = np.eye(4)
    if calib_path and yaml:
        with open(calib_path) as f:
            c = yaml.safe_load(f)
        def cam(cc):
            fx, fy, cx, cy = cc["intrinsics"]
            W, H = cc["resolution"]
            model = "equidistant" if cc.get("distortion_model", "radtan") == "equidistant" else "radtan"
            return CameraCalib(Intrinsics(fx, fy, cx, cy, W, H), np.array(cc.get("distortion_coeffs", []), float), model)
        rgb = cam(c["cam0"])
        ev_cal = cam(c["cam1"])
        if "T_cn_cnm1" in c["cam1"]:
            T = np.array(c["cam1"]["T_cn_cnm1"], float)  # cam0 -> cam1

    gt = None
    gt_path = os.path.join(d, "stamped_groundtruth.txt")
    if os.path.exists(gt_path):
        a = np.loadtxt(gt_path, comments="#", ndmin=2)
        a[:, 0] = to_seconds(a[:, 0]) - t_origin
        R_wc, t_wc = _tum_to_w2c(a)
        if gt_pose_frame == "event":  # convert event-camera poses to RGB-camera poses
            Tinv = np.linalg.inv(T)
            R_wc = np.einsum("ij,njk->nik", Tinv[:3, :3], R_wc)
            t_wc = np.einsum("ij,nj->ni", Tinv[:3, :3], t_wc) + Tinv[:3, 3]
        gt = (a[:, 0], R_wc, t_wc)

    ev_path = os.path.join(d, "events.h5")
    K_ev = ev_cal.K if ev_cal else None

    def loader(t0, t1):
        with h5py.File(ev_path, "r") as f:
            n = find_h5_events(f)
            t_all = f[n["t"]]
            ts = to_seconds(t_all[:])
            i0, i1 = np.searchsorted(ts, [t0 + t_origin, t1 + t_origin])
            ev = dict(t=ts[i0:i1] - t_origin, x=f[n["x"]][i0:i1].astype(np.int64), y=f[n["y"]][i0:i1].astype(np.int64),
                      p=np.where(f[n["p"]][i0:i1] > 0, 1, -1).astype(np.int8))
        return undistort_events(ev, ev_cal, K_ev)

    shape = (K_ev.height, K_ev.width) if K_ev else None
    return Sequence("eds", seq, frames, ft - t_origin, t_origin, rgb, ev_cal, T, gt, loader, shape,
                    notes=f"GT poses interpreted as {gt_pose_frame} camera")


# ============================================================================
# DSEC  (Gehrig et al., RA-L 2021)
# <root>/<seq>/images/left/rectified/*.png, images/timestamps.txt (us)
# <root>/<seq>/events/left/events.h5 (+ rectify_map.h5)
# <root>/<seq>/calibration/cam_to_cam.yaml
# Rendering happens in camRect0 (rectified left event camera); the frames are in camRect1.
# ============================================================================

def load_dsec(root: str, seq: str) -> Sequence:
    import h5py

    d = os.path.join(root, seq)
    frames = sorted_images(os.path.join(d, "images", "left", "rectified"))
    ft = to_seconds(np.loadtxt(os.path.join(d, "images", "timestamps.txt"), ndmin=1), "us")
    n = min(len(ft), len(frames))
    frames, ft = frames[:n], ft[:n]
    t_origin = float(ft[0])
    with open(os.path.join(d, "calibration", "cam_to_cam.yaml")) as f:
        c = yaml.safe_load(f)
    def rect(name):
        fx, fy, cx, cy = c["intrinsics"][name]["camera_matrix"]
        W, H = c["intrinsics"][name]["resolution"]
        return CameraCalib(Intrinsics(fx, fy, cx, cy, W, H), np.zeros(0))
    rgb = rect("camRect1")
    ev_cal = rect("camRect0")
    T_10 = np.array(c["extrinsics"]["T_10"], float)  # cam0 (event left) -> cam1 (rgb left)
    R0 = np.array(c["extrinsics"]["R_rect0"], float)
    R1 = np.array(c["extrinsics"]["R_rect1"], float)
    A0, A1 = np.eye(4), np.eye(4)
    A0[:3, :3], A1[:3, :3] = R0, R1
    T_r1_r0 = A1 @ T_10 @ np.linalg.inv(A0)
    T_ev_rgb = np.linalg.inv(T_r1_r0)  # camRect1 -> camRect0

    ev_path = os.path.join(d, "events", "left", "events.h5")
    rmap_path = os.path.join(d, "events", "left", "rectify_map.h5")

    def loader(t0, t1):
        with h5py.File(ev_path, "r") as f:
            t_off = float(f["t_offset"][()]) if "t_offset" in f else 0.0
            ms_to_idx = f["ms_to_idx"][:] if "ms_to_idx" in f else None
            g = f["events"]
            a0 = (t0 + t_origin) * 1e6 - t_off
            a1 = (t1 + t_origin) * 1e6 - t_off
            if ms_to_idx is not None:
                lo = int(ms_to_idx[max(0, min(len(ms_to_idx) - 1, int(a0 // 1000)))])
                hi = int(ms_to_idx[max(0, min(len(ms_to_idx) - 1, int(a1 // 1000) + 1))])
                hi = max(hi, lo)
            else:
                lo, hi = 0, len(g["t"])
            ts = g["t"][lo:hi].astype(np.float64)
            s0, s1 = np.searchsorted(ts, [a0, a1])
            ev = dict(t=(ts[s0:s1] + t_off) * 1e-6 - t_origin, x=g["x"][lo + s0:lo + s1].astype(np.int64),
                      y=g["y"][lo + s0:lo + s1].astype(np.int64), p=np.where(g["p"][lo + s0:lo + s1] > 0, 1, -1).astype(np.int8))
        with h5py.File(rmap_path, "r") as f:
            rmap = f["rectify_map"][:]  # (H, W, 2)
        xy = rmap[ev["y"], ev["x"]]
        xi, yi = np.round(xy[:, 0]).astype(np.int64), np.round(xy[:, 1]).astype(np.int64)
        K = ev_cal.K
        ok = (xi >= 0) & (xi < K.width) & (yi >= 0) & (yi < K.height)
        return dict(t=ev["t"][ok], x=xi[ok].astype(np.int32), y=yi[ok].astype(np.int32), p=ev["p"][ok])

    return Sequence("dsec", seq, frames, ft - t_origin, t_origin, rgb, ev_cal, T_ev_rgb, None, loader,
                    (ev_cal.K.height, ev_cal.K.width), notes="no GT poses: estimate with COLMAP")


# ============================================================================
# HS-ERGB  (Tulyakov et al., Time Lens, CVPR 2021)
# <root>/<close|far>/test/<seq>/images_corrected/*.png + timestamp.txt
# <root>/.../<seq>/events_aligned/*.npz  (x, y, t [us], p), already aligned to the frames
# ============================================================================

def _npz_event_folder_loader(folder, t_origin, keymap, shape, t_unit="us"):
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    # cache file start/end times lazily
    bounds = []

    def file_bounds():
        if not bounds:
            for p in files:
                with np.load(p) as z:
                    t = z[keymap["t"]]
                    if len(t):
                        b = to_seconds(np.array([t[0], t[-1]]), t_unit)
                        bounds.append((p, b[0], b[1]))
        return bounds

    def loader(t0, t1):
        a0, a1 = t0 + t_origin, t1 + t_origin
        parts = []
        for p, s, e in file_bounds():
            if e < a0 or s > a1:
                continue
            with np.load(p) as z:
                t = to_seconds(z[keymap["t"]], t_unit)
                sel = (t >= a0) & (t < a1)
                parts.append(dict(t=t[sel] - t_origin, x=z[keymap["x"]][sel].astype(np.int32),
                                  y=z[keymap["y"]][sel].astype(np.int32),
                                  p=np.where(z[keymap["p"]][sel] > 0, 1, -1).astype(np.int8)))
        if not parts:
            return dict(t=np.zeros(0), x=np.zeros(0, np.int32), y=np.zeros(0, np.int32), p=np.zeros(0, np.int8))
        ev = {k: np.concatenate([q[k] for q in parts]) for k in ("t", "x", "y", "p")}
        o = np.argsort(ev["t"], kind="stable")
        ev = {k: v[o] for k, v in ev.items()}
        H, W = shape
        ok = (ev["x"] >= 0) & (ev["x"] < W) & (ev["y"] >= 0) & (ev["y"] < H)
        return {k: v[ok] for k, v in ev.items()}

    return loader


def load_hsergb(root: str, seq: str) -> Sequence:
    from PIL import Image

    d = os.path.join(root, seq)
    img_dir = os.path.join(d, "images_corrected")
    frames = sorted_images(img_dir)
    ts_path = next(p for p in [os.path.join(img_dir, "timestamp.txt"), os.path.join(img_dir, "timestamps.txt")] if os.path.exists(p))
    ft = to_seconds(np.loadtxt(ts_path, ndmin=1))
    n = min(len(ft), len(frames))
    frames, ft = frames[:n], ft[:n]
    W, H = Image.open(frames[0]).size
    t_origin = float(ft[0])
    loader = _npz_event_folder_loader(os.path.join(d, "events_aligned"), t_origin, dict(t="t", x="x", y="y", p="p"), (H, W))
    return Sequence("hsergb", seq, frames, ft - t_origin, t_origin, None, None, np.eye(4), None, loader, (H, W),
                    notes="events pre-aligned to frames; intrinsics + poses from Easi3R/COLMAP")


# ============================================================================
# BS-ERGB  (Tulyakov et al., Time Lens++, CVPR 2022)
# <root>/<split>/<seq>/images/*.png + images/timestamps.txt
# <root>/<split>/<seq>/events/*.npz  (x, y, timestamp [us], polarity), aligned to the frames
# ============================================================================

def load_bsergb(root: str, seq: str) -> Sequence:
    from PIL import Image

    d = os.path.join(root, seq)
    img_dir = os.path.join(d, "images")
    frames = sorted_images(img_dir)
    ts_path = next(p for p in [os.path.join(img_dir, "timestamps.txt"), os.path.join(d, "timestamps.txt")] if os.path.exists(p))
    ft = to_seconds(np.loadtxt(ts_path, ndmin=1))
    n = min(len(ft), len(frames))
    frames, ft = frames[:n], ft[:n]
    W, H = Image.open(frames[0]).size
    t_origin = float(ft[0])
    sample = sorted(glob.glob(os.path.join(d, "events", "*.npz")))
    keys = dict(t="timestamp", x="x", y="y", p="polarity")
    if sample:
        with np.load(sample[0]) as z:
            files = set(z.files)
        if "t" in files:
            keys["t"] = "t"
        if "p" in files:
            keys["p"] = "p"
    loader = _npz_event_folder_loader(os.path.join(d, "events"), t_origin, keys, (H, W))
    return Sequence("bsergb", seq, frames, ft - t_origin, t_origin, None, None, np.eye(4), None, loader, (H, W),
                    notes="events pre-aligned to frames; intrinsics + poses from Easi3R/COLMAP")


LOADERS = {"eds": load_eds, "dsec": load_dsec, "hsergb": load_hsergb, "bsergb": load_bsergb}


def load_sequence(dataset: str, root: str, seq: str, **kw) -> Sequence:
    return LOADERS[dataset](root, seq, **kw)
