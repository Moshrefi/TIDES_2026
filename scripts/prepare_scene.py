#!/usr/bin/env python
"""Prepare one sequence for 4DGS training and TIDES simulation.

Writes <out>/scene (COLMAP-text layout that the 4DGS repo reads) and
<out>/tides_meta.json.

Pose / intrinsics sources, in priority order:
  1. --colmap <sparse model dir>  (cameras + images [+ points3D], .txt or .bin), e.g.
     from COLMAP on static scenes or exported from Easi3R for dynamic scenes
  2. --poses <TUM file> (timestamp tx ty tz qx qy qz qw, camera-to-world, RGB camera)
  3. the dataset's own ground truth (EDS)

Examples:
  python scripts/prepare_scene.py --dataset eds --root /data/EDS --seq 00_peanuts_dark \
      --out work/eds/00_peanuts_dark --t-start 0 --t-end 4
  python scripts/prepare_scene.py --dataset hsergb --root /data/hsergb/close/test --seq spinning_umbrella \
      --colmap /data/easi3r_out/spinning_umbrella/sparse/0 --out work/hsergb/spinning_umbrella
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tides.camera import Intrinsics, PoseTrajectory, quat_to_rotmat, rotmat_to_quat  # noqa: E402
from tides.datasets import load_sequence  # noqa: E402
from tides.datasets.base import to_seconds  # noqa: E402


def read_colmap_model(path):
    """Minimal COLMAP model reader (text or binary via pycolmap if installed)."""
    cams, imgs, pts = {}, {}, None
    if os.path.exists(os.path.join(path, "cameras.txt")):
        for line in open(os.path.join(path, "cameras.txt")):
            if line.startswith("#") or not line.strip():
                continue
            e = line.split()
            cid, model, W, H = int(e[0]), e[1], int(e[2]), int(e[3])
            p = list(map(float, e[4:]))
            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = p[0]; cx, cy = p[1], p[2]; dist = p[3:] + [0.0] * (4 - len(p[3:]))
                dist = [dist[0], dist[1] if model == "RADIAL" else 0.0, 0.0, 0.0]
            else:  # PINHOLE, OPENCV
                fx, fy, cx, cy = p[:4]; dist = p[4:8] if model == "OPENCV" else []
            cams[cid] = (Intrinsics(fx, fy, cx, cy, W, H), np.array(dist, float))
        lines = [l for l in open(os.path.join(path, "images.txt")) if not l.startswith("#")]
        for i in range(0, len(lines), 2):
            e = lines[i].split()
            if len(e) < 10:
                continue
            q = np.array(list(map(float, e[1:5])))
            t = np.array(list(map(float, e[5:8])))
            imgs[e[9]] = (quat_to_rotmat(__import__("torch").tensor(q, dtype=__import__("torch").float64)).numpy(), t, int(e[8]))
        if os.path.exists(os.path.join(path, "points3D.txt")):
            P = []
            for line in open(os.path.join(path, "points3D.txt")):
                if line.startswith("#") or not line.strip():
                    continue
                e = line.split()
                P.append(list(map(float, e[1:7])))
            pts = np.array(P) if P else None
    else:
        import pycolmap

        rec = pycolmap.Reconstruction(path)
        for cid, c in rec.cameras.items():
            p = c.params
            if c.model.name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                K = Intrinsics(p[0], p[0], p[1], p[2], c.width, c.height); dist = [p[3], 0, 0, 0] if len(p) > 3 else []
            else:
                K = Intrinsics(p[0], p[1], p[2], p[3], c.width, c.height); dist = list(p[4:8])
            cams[cid] = (K, np.array(dist, float))
        for _, im in rec.images.items():
            T = im.cam_from_world.matrix()
            imgs[im.name] = (np.asarray(T[:3, :3]), np.asarray(T[:3, 3]), im.camera_id)
        pts = np.array([[*p.xyz, *p.color] for p in rec.points3D.values()]) if rec.points3D else None
    return cams, imgs, pts


def write_ply(path, xyz, rgb):
    """Binary PLY in the layout 3DGS/4DGS storePly() writes (x y z nx ny nz red green blue)."""
    n = len(xyz)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v = np.zeros(n, dtype=dt)
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    c = np.clip(rgb, 0, 255).astype(np.uint8)
    v["red"], v["green"], v["blue"] = c[:, 0], c[:, 1], c[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(v.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["eds", "dsec", "hsergb", "bsergb"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--colmap", default=None)
    ap.add_argument("--poses", default=None, help="TUM file (c2w) for the RGB camera")
    ap.add_argument("--intrinsics", default=None, help="fx,fy,cx,cy of the RGB frames if the dataset has none")
    ap.add_argument("--t-start", type=float, default=0.0, help="seconds from first frame")
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--n-random-points", type=int, default=100_000)
    ap.add_argument("--depth-range", type=float, nargs=2, default=(0.5, 8.0))
    ap.add_argument("--target-depth", type=float, default=4.0,
                    help="rescale the scene so the median point depth is this (needs points); 0 = keep scale")
    ap.add_argument("--gt-pose-frame", default="rgb", choices=["rgb", "event"])
    ap.add_argument("--background", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                    help="must match the 4DGS training background (4DGS defaults to white)")
    args = ap.parse_args()

    import cv2

    kw = {"gt_pose_frame": args.gt_pose_frame} if args.dataset == "eds" else {}
    seq = load_sequence(args.dataset, args.root, args.seq, **kw)
    ft = seq.frame_times
    t_end = args.t_end if args.t_end is not None else ft[-1]
    sel = np.nonzero((ft >= args.t_start) & (ft <= t_end))[0][:: args.stride][: args.max_frames]
    assert len(sel) >= 2, "need at least two frames in the selected range"
    print(f"[prepare] {seq.dataset}/{seq.name}: {len(sel)} frames, {ft[sel[0]]:.3f}s .. {ft[sel[-1]]:.3f}s")

    # ---- intrinsics and poses ------------------------------------------------
    pts = None
    names = [os.path.basename(seq.frame_paths[i]) for i in sel]
    if args.colmap:
        cams, imgs, pts = read_colmap_model(args.colmap)
        K_rgb, dist = cams[next(iter(cams))]
        R_list, t_list, keep = [], [], []
        for j, n in enumerate(names):
            if n in imgs:
                R_list.append(imgs[n][0]); t_list.append(imgs[n][1]); keep.append(j)
        assert len(keep) >= 2, "COLMAP model does not contain the selected frames (matched by file name)"
        sel = sel[keep]
        R_wc, t_wc = np.stack(R_list), np.stack(t_list)
        print(f"[prepare] poses from COLMAP model: {len(keep)} frames registered")
    else:
        if seq.rgb is not None:
            K_rgb, dist = seq.rgb.K, seq.rgb.dist
        else:
            assert args.intrinsics, "this dataset has no calibration: pass --intrinsics or --colmap"
            from PIL import Image

            W, H = Image.open(seq.frame_paths[0]).size
            fx, fy, cx, cy = map(float, args.intrinsics.split(","))
            K_rgb, dist = Intrinsics(fx, fy, cx, cy, W, H), np.zeros(0)
        if args.poses:
            a = np.loadtxt(args.poses, comments="#", ndmin=2)
            a[:, 0] = to_seconds(a[:, 0])
            if a[0, 0] > 1e3:
                a[:, 0] -= seq.t_origin
            from tides.datasets.loaders import _tum_to_w2c

            R_all, t_all = _tum_to_w2c(a)
            gt = (a[:, 0], R_all, t_all)
        else:
            assert seq.gt_poses is not None, "no poses: pass --poses or --colmap"
            gt = seq.gt_poses
        traj = PoseTrajectory(*gt, device="cpu")
        import torch

        R_wc, t_wc, _ = traj.pose(torch.tensor(ft[sel], dtype=torch.float32))
        R_wc, t_wc = R_wc.double().numpy(), t_wc.double().numpy()
        lo, hi = gt[0][0], gt[0][-1]
        inside = (ft[sel] >= lo) & (ft[sel] <= hi)
        if not inside.all():
            print(f"[prepare] dropping {np.sum(~inside)} frames outside the ground-truth time range")
            sel, R_wc, t_wc = sel[inside], R_wc[inside], t_wc[inside]

    # ---- undistort + centre the principal point (4DGS assumes cx = (W-1)/2) ---
    W, H = K_rgb.width, K_rgb.height
    hx = int(np.floor(min(K_rgb.cx, W - 1 - K_rgb.cx)))
    hy = int(np.floor(min(K_rgb.cy, H - 1 - K_rgb.cy)))
    Wc, Hc = 2 * hx + 1, 2 * hy + 1
    K_c = Intrinsics(K_rgb.fx, K_rgb.fy, (Wc - 1) / 2, (Hc - 1) / 2, Wc, Hc)
    Kin = np.array([[K_rgb.fx, 0, K_rgb.cx], [0, K_rgb.fy, K_rgb.cy], [0, 0, 1]])
    Kout = np.array([[K_c.fx, 0, K_c.cx], [0, K_c.fy, K_c.cy], [0, 0, 1]])
    d = dist if dist is not None and len(dist) else np.zeros(4)
    if seq.rgb is not None and seq.rgb.model == "equidistant":
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(Kin, d[:4], np.eye(3), Kout, (Wc, Hc), cv2.CV_32FC1)
    else:
        m1, m2 = cv2.initUndistortRectifyMap(Kin, d, None, Kout, (Wc, Hc), cv2.CV_32FC1)

    out_scene = os.path.join(args.out, "scene")
    os.makedirs(os.path.join(out_scene, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_scene, "sparse", "0"), exist_ok=True)
    imgs_rgb = []
    for k, i in enumerate(sel):
        im = cv2.imread(seq.frame_paths[i], cv2.IMREAD_COLOR)
        if im.shape[1] != W or im.shape[0] != H:
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        und = cv2.remap(im, m1, m2, cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(out_scene, "images", f"{k:06d}.png"), und)
        if k % max(1, len(sel) // 20) == 0:
            imgs_rgb.append((k, und[:, :, ::-1]))

    # ---- points: COLMAP points or random points in the camera frusta ----------
    rng = np.random.default_rng(0)
    if pts is None or len(pts) < 100:
        n = args.n_random_points
        P = []
        per = max(1, n // len(imgs_rgb))
        for k, img in imgs_rgb:
            u = rng.uniform(0, Wc - 1, per); v = rng.uniform(0, Hc - 1, per)
            z = rng.uniform(*args.depth_range, per)
            xc = np.stack([(u - K_c.cx) / K_c.fx * z, (v - K_c.cy) / K_c.fy * z, z], -1)
            Rw, tw = R_wc[k], t_wc[k]
            xw = (xc - tw) @ Rw  # R^T (x - t)
            col = img[v.astype(int), u.astype(int)]
            P.append(np.concatenate([xw, col], -1))
        pts = np.concatenate(P)
        print(f"[prepare] random point initialisation: {len(pts)} points")

    # ---- optional rescale -----------------------------------------------------
    scale = 1.0
    if args.target_depth > 0 and args.colmap:
        depths = []
        for k in range(0, len(sel), max(1, len(sel) // 10)):
            z = (pts[:, :3] @ R_wc[k].T + t_wc[k])[:, 2]
            depths.append(z[z > 0])
        med = np.median(np.concatenate(depths))
        scale = args.target_depth / med
        pts[:, :3] *= scale
        t_wc = t_wc * scale
        print(f"[prepare] rescaled scene by {scale:.4f} (median depth {med:.3f} -> {args.target_depth})")

    # ---- COLMAP text model -----------------------------------------------------
    with open(os.path.join(out_scene, "sparse", "0", "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {Wc} {Hc} {K_c.fx} {K_c.fy} {K_c.cx} {K_c.cy}\n")
    with open(os.path.join(out_scene, "sparse", "0", "images.txt"), "w") as f:
        for k in range(len(sel)):
            q = rotmat_to_quat(R_wc[k])
            t = t_wc[k]
            f.write(f"{k + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {k:06d}.png\n\n")
    with open(os.path.join(out_scene, "sparse", "0", "points3D.txt"), "w") as f:
        for i, p in enumerate(pts):
            f.write(f"{i + 1} {p[0]} {p[1]} {p[2]} {int(p[3])} {int(p[4])} {int(p[5])} 0\n")
    # 4DGS converts points3D.txt to .ply with a quadratic-time loop; write the .ply directly
    write_ply(os.path.join(out_scene, "sparse", "0", "points3D.ply"), pts[:, :3], pts[:, 3:6])

    # ---- render camera (event camera, undistorted) ------------------------------
    T = np.array(seq.T_ev_rgb, float).copy()
    T[:3, 3] *= scale
    if seq.event is not None:
        K_r = seq.event.K
    else:  # events are aligned to the RGB frames in the original pixel grid
        K_r = K_rgb
    frame_times = ft[sel]
    N = len(sel)
    meta = dict(
        dataset=seq.dataset, sequence=seq.name, root=os.path.abspath(args.root), t_origin=seq.t_origin,
        frame_times_s=frame_times.tolist(), scene_times=[k / N for k in range(N)],
        frame_interval=float(np.median(np.diff(frame_times))),
        t_range=[float(frame_times[0]), float(frame_times[-1])],
        K_train=K_c.as_dict(), K_render=K_r.as_dict(), T_render_rgb=T.tolist(), scale=scale,
        poses=dict(times_s=frame_times.tolist(), R_wc=R_wc.tolist(), t_wc=t_wc.tolist()),
        background=list(args.background), notes=seq.notes,
        loader_kwargs=kw,
    )
    with open(os.path.join(args.out, "tides_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[prepare] wrote {args.out}  (train {Wc}x{Hc}, render {K_r.width}x{K_r.height})")


if __name__ == "__main__":
    main()
