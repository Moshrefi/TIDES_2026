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


def _layout_error(d, what):
    """Readable error showing what is actually in the sequence folder."""
    lines = [f"{what}. Contents of {d}:"]
    for name in sorted(os.listdir(d))[:40]:
        p = os.path.join(d, name)
        if os.path.isdir(p):
            sub = sorted(os.listdir(p))
            lines.append(f"  {name}/  ({len(sub)} entries, e.g. {sub[:3]})")
        else:
            lines.append(f"  {name}")
    lines.append("Please send this listing so the loader can be adapted.")
    return "\n".join(lines)


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

def load_npy_layout(root: str, seq: str, dataset: str = "npy") -> Sequence:
    """Pre-processed layout (e.g. the EvTween copy of EDS):

    <seq>/frames/*.png, frame_ts.npy, ev_t.npy, ev_x.npy, ev_y.npy, ev_p.npy [, align.yaml]

    Events are assumed to be already aligned to the frames' pixel grid. There
    are no poses or intrinsics: pass --colmap (or --intrinsics + --poses) to
    prepare_scene.py.
    """
    from PIL import Image

    d = os.path.join(root, seq)
    img_dir = next((p for p in [os.path.join(d, "frames"), os.path.join(d, "images")] if os.path.isdir(p)), None)
    if img_dir is None:
        raise FileNotFoundError(_layout_error(d, "no frames/ folder"))
    frames = sorted_images(img_dir)
    ft = to_seconds(np.load(os.path.join(d, "frame_ts.npy")).reshape(-1))
    n = min(len(ft), len(frames))
    if len(ft) != len(frames):
        print(f"[npy loader] {len(ft)} timestamps vs {len(frames)} frames; using the first {n}")
    frames, ft = frames[:n], ft[:n]
    t_origin = float(ft[0])
    W, H = Image.open(frames[0]).size
    align = None
    if os.path.exists(os.path.join(d, "align.yaml")) and yaml:
        with open(os.path.join(d, "align.yaml")) as f:
            align = yaml.safe_load(f)
    ev_t_raw = np.load(os.path.join(d, "ev_t.npy"), mmap_mode="r")
    # decide the event time unit once, from the span relative to the frames
    span = float(ev_t_raw[-1]) - float(ev_t_raw[0])
    frame_span = ft[-1] - ft[0]
    scale = 1.0
    for s_ in (1.0, 1e-3, 1e-6, 1e-9):
        if 0.2 * frame_span <= span * s_ <= 5 * frame_span + 1.0:
            scale = s_
            break
    ev_off = 0.0
    # events may share the frames' clock or start at zero
    if abs(float(ev_t_raw[0]) * scale - t_origin) > 10 * frame_span and abs(float(ev_t_raw[0]) * scale) < frame_span:
        ev_off = t_origin
        print("[npy loader] event timestamps look relative; assuming they start with the first frame")

    def loader(t0, t1):
        tt = np.load(os.path.join(d, "ev_t.npy"), mmap_mode="r")
        a0 = (t0 + t_origin - ev_off) / scale
        a1 = (t1 + t_origin - ev_off) / scale
        i0, i1 = np.searchsorted(tt, [a0, a1])
        t = np.asarray(tt[i0:i1], dtype=np.float64) * scale + ev_off - t_origin
        x = np.asarray(np.load(os.path.join(d, "ev_x.npy"), mmap_mode="r")[i0:i1]).astype(np.int32)
        y = np.asarray(np.load(os.path.join(d, "ev_y.npy"), mmap_mode="r")[i0:i1]).astype(np.int32)
        p = np.where(np.asarray(np.load(os.path.join(d, "ev_p.npy"), mmap_mode="r")[i0:i1]) > 0, 1, -1).astype(np.int8)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        return dict(t=t[ok], x=x[ok], y=y[ok], p=p[ok])

    return Sequence(dataset, seq, frames, ft - t_origin, t_origin, None, None, np.eye(4), None, loader, (H, W),
                    notes=f"npy layout; events assumed aligned to frames; event time scale {scale}; align.yaml={align}")


def load_eds(root: str, seq: str, gt_pose_frame: str = "rgb", calib_path: str | None = None) -> Sequence:
    import h5py

    if gt_pose_frame not in ("rgb", "event"):
        raise ValueError("gt_pose_frame must be rgb or event")
    d = os.path.join(root, seq)
    if os.path.exists(os.path.join(d, "frame_ts.npy")) and os.path.exists(os.path.join(d, "ev_t.npy")):
        s = load_npy_layout(root, seq, dataset="eds")
        print("[eds] pre-processed .npy layout detected (no calib.yaml / ground-truth poses here)")
        return s
    if not os.path.isdir(d):
        raise FileNotFoundError(f"EDS sequence folder not found: {d}")
    img_dir = next((p for p in [os.path.join(d, "images"), os.path.join(d, "frames"), d] if os.path.isdir(p) and sorted_images(p)), None)
    if img_dir is None:
        raise FileNotFoundError(_layout_error(d, "no image folder (expected <seq>/images/*.png)"))
    frames = sorted_images(img_dir)
    ts_cands = [os.path.join(d, "images_timestamps.txt"), os.path.join(img_dir, "timestamps.txt")]
    ts_cands += sorted(glob.glob(os.path.join(d, "*timestamp*.txt")) + glob.glob(os.path.join(img_dir, "*timestamp*.txt")))
    ts_file = next((p for p in ts_cands if os.path.exists(p)), None)
    if ts_file is None:
        raise FileNotFoundError(_layout_error(d, "no frame timestamp file (expected <seq>/images_timestamps.txt)"))
    ft = to_seconds(_read_txt_column(ts_file, -1 if np.loadtxt(ts_file, ndmin=2).shape[1] > 1 else 0))
    assert len(ft) == len(frames), f"{len(ft)} timestamps vs {len(frames)} images"
    t_origin = float(ft[0])

    calib_cands = [os.path.join(d, "calib.yaml"), os.path.join(root, "calib.yaml")]
    for base in (d, root, os.path.dirname(os.path.abspath(root))):
        calib_cands += sorted(glob.glob(os.path.join(base, "*camchain*.yaml")) + glob.glob(os.path.join(base, "*calib*", "*camchain*.yaml")))
        calib_cands += sorted(glob.glob(os.path.join(base, "calib", "00_calib", "*camchain*.yaml")))
    explicit = calib_path or os.environ.get("TIDES_EDS_CALIB")
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"EDS calibration explicitly selected but missing: {explicit}")
        calib_path = os.path.abspath(explicit)
    else:
        found = sorted({os.path.realpath(p) for p in calib_cands if os.path.isfile(p)})
        if len(found) > 1:
            raise ValueError(f"Ambiguous EDS calibration; pass calib_path / --calib explicitly: {found}")
        calib_path = found[0] if found else None
    if calib_path:
        print(f"[eds] calibration: {calib_path}")
    rgb = ev_cal = None
    T = np.eye(4)
    if calib_path and yaml is None:
        raise ImportError("Reading EDS calibration requires PyYAML")
    if calib_path:
        with open(calib_path) as f:
            c = yaml.safe_load(f)
        def cam(cc):
            if cc.get("camera_model", "pinhole") != "pinhole":
                raise ValueError("Only pinhole EDS calibration is supported")
            if cc.get("distortion_model", "radtan") not in ("radtan", "equidistant"):
                raise ValueError("Unsupported EDS distortion_model")
            fx, fy, cx, cy = cc["intrinsics"]
            W, H = cc["resolution"]
            model = "equidistant" if cc.get("distortion_model", "radtan") == "equidistant" else "radtan"
            return CameraCalib(Intrinsics(fx, fy, cx, cy, W, H), np.array(cc.get("distortion_coeffs", []), float), model)
        rgb = cam(c["cam0"])
        ev_cal = cam(c["cam1"])
        if "T_cn_cnm1" in c["cam1"]:
            T = np.array(c["cam1"]["T_cn_cnm1"], float)  # cam0 -> cam1
        else:
            raise ValueError("EDS cam1 calibration is missing T_cn_cnm1 (RGB -> event)")
        if (T.shape != (4, 4) or not np.isfinite(T).all()
                or not np.allclose(T[3], [0, 0, 0, 1])
                or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5)):
            raise ValueError("EDS T_cn_cnm1 must be a rigid 4x4 transform")
    if gt_pose_frame == "event" and calib_path is None:
        raise ValueError("Converting event GT poses to RGB requires calibration")

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
    if not os.path.exists(ev_path):
        h5s = sorted(glob.glob(os.path.join(d, "*.h5")) + glob.glob(os.path.join(d, "*", "*.h5")))
        if not h5s:
            raise FileNotFoundError(_layout_error(d, "no events .h5 file (expected <seq>/events.h5)"))
        ev_path = h5s[0]
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
                    notes=f"GT poses interpreted as {gt_pose_frame} camera", calibration_path=calib_path)


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
