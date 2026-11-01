"""Integration test: a fake EDS-style sequence through prepare_scene + dataset loader."""
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import torch
import yaml
from PIL import Image

from tides.camera import quat_to_rotmat
from tides.datasets import load_sequence

ROOT = os.path.join(os.path.dirname(__file__), "..")


def make_fake_eds(root, n=12, W=64, H=48):
    seq = os.path.join(root, "fake_seq")
    os.makedirs(os.path.join(seq, "images"))
    t_us = 1_600_000_000_000_000 // 1000 + np.arange(n) * 33_333  # absolute microseconds
    for i in range(n):
        img = (np.random.default_rng(i).random((H, W, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(seq, "images", f"frame_{i:010d}.png"))
    np.savetxt(os.path.join(seq, "images_timestamps.txt"), t_us, fmt="%d")
    # ground truth: camera moving along x, identity orientation (TUM: t tx ty tz qx qy qz qw)
    gt_t = np.linspace(t_us[0], t_us[-1], 50) * 1e-6
    gt = np.stack([gt_t, np.linspace(0, 1, 50), np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50), np.ones(50)], -1)
    np.savetxt(os.path.join(seq, "stamped_groundtruth.txt"), gt)
    calib = {"cam0": {"intrinsics": [50.0, 50.0, 31.0, 24.0], "resolution": [W, H], "distortion_coeffs": [0.0, 0, 0, 0]},
             "cam1": {"intrinsics": [50.0, 50.0, 32.0, 23.0], "resolution": [W, H], "distortion_coeffs": [0.0, 0, 0, 0],
                      "T_cn_cnm1": np.eye(4).tolist()}}
    yaml.safe_dump(calib, open(os.path.join(root, "calib.yaml"), "w"))
    m = 5000
    with h5py.File(os.path.join(seq, "events.h5"), "w") as f:
        f["t"] = np.sort(np.random.default_rng(0).integers(t_us[0], t_us[-1], m))
        f["x"] = np.random.default_rng(1).integers(0, W, m)
        f["y"] = np.random.default_rng(2).integers(0, H, m)
        f["p"] = np.random.default_rng(3).integers(0, 2, m)
    return "fake_seq"


def test_prepare_scene_fake_eds(tmp_path):
    root = str(tmp_path / "eds")
    os.makedirs(root)
    seq = make_fake_eds(root)
    s = load_sequence("eds", root, seq)
    assert len(s.frame_paths) == 12 and abs(s.frame_times[1] - 0.033333) < 1e-6
    ev = s.load_events(0.0, 0.2)
    assert len(ev["t"]) > 0 and ev["t"].min() >= 0 and ev["t"].max() < 0.2

    out = str(tmp_path / "work")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "prepare_scene.py"), "--dataset", "eds", "--root", root,
                    "--seq", seq, "--out", out, "--n-random-points", "500"], check=True)
    meta = json.load(open(os.path.join(out, "tides_meta.json")))
    assert len(os.listdir(os.path.join(out, "scene", "images"))) == len(meta["frame_times_s"])
    assert os.path.exists(os.path.join(out, "scene", "sparse", "0", "points3D.ply"))
    # principal point centred for 4DGS
    K = meta["K_train"]
    assert abs(K["cx"] - (K["width"] - 1) / 2) < 1e-9 and abs(K["cy"] - (K["height"] - 1) / 2) < 1e-9
    # images.txt pose of frame k must reproduce the GT camera centre (x moves 0 -> 1 over the GT span)
    lines = [l for l in open(os.path.join(out, "scene", "sparse", "0", "images.txt")) if l.strip()]
    e = lines[-1].split()
    R = quat_to_rotmat(torch.tensor(list(map(float, e[1:5])), dtype=torch.float64)).numpy()
    t = np.array(list(map(float, e[5:8])))
    centre = -R.T @ t
    t_last = meta["frame_times_s"][-1]
    gt_span = 11 * 0.033333
    assert abs(centre[0] - t_last / gt_span) < 1e-3


def test_rotated_tum_and_event_to_rgb_composition():
    """Use nonidentity rotations/translations so an inverse or composition error cannot hide."""
    from tides.datasets.loaders import _tum_to_w2c
    from tides.workdir import render_trajectory
    a = np.array([[0, 1, 2, 3, 0, 0, np.sqrt(.5), np.sqrt(.5)]])
    R, t = _tum_to_w2c(a)
    np.testing.assert_allclose(-R[0].T @ t[0], [1,2,3], atol=1e-12)
    T = np.eye(4)
    T[:3,:3] = np.array([[0,0,1],[0,1,0],[-1,0,0]])
    T[:3,3] = [.1,.2,.3]
    meta = dict(poses=dict(times_s=[0], R_wc=R.tolist(), t_wc=t.tolist()), T_render_rgb=T.tolist())
    traj = render_trajectory(meta)
    Rev, tev, _ = traj.pose(torch.tensor([0.]))
    expected = T @ np.block([[R[0], t[0,:,None]], [np.zeros((1,3)), np.ones((1,1))]])
    np.testing.assert_allclose(Rev.numpy()[0], expected[:3,:3], atol=1e-6)
    np.testing.assert_allclose(tev.numpy()[0], expected[:3,3], atol=1e-6)


def test_time_map_irregular_seconds_and_derivative():
    from tides.scene.fourdgs import TimeMap
    tm = TimeMap([0,.01,.029], [0,1/3,2/3])
    t = torch.tensor([0,.005,.01,.029], requires_grad=True)
    st = tm(t)
    torch.testing.assert_close(st, torch.tensor([0,1/6,1/3,2/3]))
    st.sum().backward()
    torch.testing.assert_close(t.grad, torch.tensor([1/.03,1/.03,(1/3)/.019,(1/3)/.019]))


def test_calibration_selection_persisted_and_existing_scene_protected(tmp_path):
    root = str(tmp_path/'eds')
    os.makedirs(root)
    seq = make_fake_eds(root)
    calibration = str(tmp_path/'eds'/'calib.yaml')
    import pytest
    with pytest.raises(FileNotFoundError, match='explicitly selected'):
        load_sequence('eds', root, seq, calib_path=str(tmp_path/'missing.yaml'))
    out = str(tmp_path/'work')
    cmd = [sys.executable, os.path.join(ROOT,'scripts','prepare_scene.py'),'--dataset','eds','--root',root,
           '--seq',seq,'--out',out,'--calib',calibration,'--n-random-points','100']
    subprocess.run(cmd, check=True)
    meta = json.load(open(os.path.join(out,'tides_meta.json')))
    assert meta['loader_kwargs']['calib_path'] == os.path.abspath(calibration)
    assert len(meta['calibration_sha256']) == 64
    before = open(os.path.join(out,'tides_meta.json'),'rb').read()
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode != 0
    assert open(os.path.join(out,'tides_meta.json'),'rb').read() == before
