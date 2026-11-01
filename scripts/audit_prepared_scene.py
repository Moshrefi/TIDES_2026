#!/usr/bin/env python
"""Read-only CPU audit of prepared COLMAP geometry/timing; never trains a model."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rotation(q):
    q = np.asarray(q, float)
    require(np.isfinite(q).all() and abs(np.linalg.norm(q) - 1) < 1e-5, "Invalid COLMAP quaternion")
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def rigid(T, name):
    require(T.shape[-2:] == (4, 4) and np.isfinite(T).all(), f"Invalid {name}")
    require(np.allclose(T[..., 3, :], [0, 0, 0, 1]), f"Invalid {name} bottom row")
    R = T[..., :3, :3]
    require(np.allclose(np.swapaxes(R, -1, -2) @ R, np.eye(3), atol=1e-5)
            and np.allclose(np.linalg.det(R), 1, atol=1e-5), f"Nonrigid {name}")


def revision(path):
    r = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def audit(work, calib=None):
    work = Path(work)
    meta = json.loads((work / 'tides_meta.json').read_text(encoding="utf-8"))
    ft = np.asarray(meta['frame_times_s'], float)
    st = np.asarray(meta['scene_times'], float)
    n = len(ft)
    require(n >= 2 and ft.ndim == 1 and np.isfinite(ft).all() and np.all(np.diff(ft) > 0),
            'Frame timestamps must be finite and strictly increasing')
    require(st.shape == ft.shape and np.allclose(st, np.arange(n)/n, atol=1e-7, rtol=0),
            'Scene times must match upstream COLMAP index/N (not seconds or index/(N-1))')
    require(np.allclose(meta['t_range'], [ft[0], ft[-1]]), 't_range does not match selected frames')
    require(np.isclose(meta['frame_interval'], np.median(np.diff(ft))), 'Incorrect frame_interval')
    require(np.allclose(meta['poses']['times_s'], ft), 'Pose/frame times differ')
    R = np.asarray(meta['poses']['R_wc'], float)
    t = np.asarray(meta['poses']['t_wc'], float)
    require(R.shape == (n, 3, 3) and t.shape == (n, 3), 'Pose array shapes differ from frame count')
    A = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    A[:, :3, :3], A[:, :3, 3] = R, t
    rigid(A, 'RGB world-to-camera poses')
    T = np.asarray(meta['T_render_rgb'], float)
    rigid(T, 'RGB-to-event transform')
    scale = float(meta['scale'])
    require(np.isfinite(scale) and scale > 0, 'Invalid scale')
    sparse = work / 'scene' / 'sparse' / '0'
    require(not any((sparse / p).exists() for p in ['images.bin', 'cameras.bin']),
            'Stale binary COLMAP model overrides the text model; use a fresh preparation output')
    cam = [l.split() for l in (sparse/'cameras.txt').read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith('#')]
    require(len(cam) == 1 and cam[0][0:2] == ['1', 'PINHOLE'], 'Expected one PINHOLE training camera')
    K = meta['K_train']
    require(np.allclose(np.array(cam[0][2:], float), [K[k] for k in ['width','height','fx','fy','cx','cy']]),
            'COLMAP intrinsics differ from metadata')
    require(np.isclose(K['cx'], (K['width']-1)/2) and np.isclose(K['cy'], (K['height']-1)/2),
            'Training principal point is not centred')
    names = []
    lines = [l for l in (sparse/'images.txt').read_text(encoding="utf-8").splitlines() if not l.startswith('#')]
    require(len(lines) == 2*n, 'Expected exactly two COLMAP text lines per selected image')
    for i in range(n):
        e = lines[2*i].split()
        require(len(e) == 10 and e[8] == '1' and int(e[0]) == i+1, 'Invalid COLMAP image record')
        require(e[9] == f'{i:06d}.png', 'COLMAP enumeration must match chronological frame order')
        require(np.allclose(rotation([float(x) for x in e[1:5]]), R[i], atol=1e-5)
                and np.allclose(np.array(e[5:8], float), t[i], atol=1e-5),
                f'COLMAP pose differs from metadata for frame {i}')
        names.append(e[9])
    actual = sorted(p.name for p in (work/'scene'/'images').iterdir() if p.is_file())
    require(actual == names, 'Missing or stale training images; use a fresh preparation output')
    require((sparse/'points3D.ply').is_file(), 'Missing initial point cloud')
    warnings = ['Algebraic consistency does not establish which physical camera the GT file describes. '
                'gt_pose_frame remains an explicit dataset assumption until the archive convention is verified.']
    selected = calib or meta.get('loader_kwargs', {}).get('calib_path')
    if selected:
        import yaml
        selected = Path(selected).resolve()
        digest = hashlib.sha256(selected.read_bytes()).hexdigest()
        old = meta.get('calibration_sha256')
        require(old is None or old == digest, 'Calibration content changed since preparation')
        c = yaml.safe_load(selected.read_text(encoding="utf-8"))
        if c['cam0'].get('flip'):
            warnings.append('Calibration marks cam0 flip=True. Confirm whether the archive RGB images are '
                            'already flipped before training; the current loader preserves archive pixels.')
        require(np.allclose(c['cam1']['intrinsics'], [meta['K_render'][k] for k in ['fx','fy','cx','cy']])
                and list(c['cam1']['resolution']) == [meta['K_render']['width'], meta['K_render']['height']],
                'Event calibration differs from prepared render camera')
        Tc = np.array(c['cam1']['T_cn_cnm1'], float)
        Tc[:3, 3] *= scale
        require(np.allclose(Tc, T, atol=1e-7), 'Calibration RGB-to-event direction/value differs from metadata')
        fx, fy, cx, cy = c['cam0']['intrinsics']
        w, h = c['cam0']['resolution']
        hx, hy = np.floor(min(cx, w-1-cx)), np.floor(min(cy, h-1-cy))
        require(np.allclose([K[k] for k in ['fx','fy','cx','cy','width','height']],
                            [fx,fy,hx,hy,2*hx+1,2*hy+1]), 'RGB crop/intrinsics differ from calibration')
        if old is None:
            warnings.append('Legacy metadata has no calibration hash; camera values match the explicitly supplied YAML.')
    else:
        digest = None
        warnings.append('No calibration file recorded; supply --calib to cross-check the existing scene.')
    return dict(status='passed', frames=n, t_range=meta['t_range'], scene_time_last=float(st[-1]),
                calibration_sha256=digest, warnings=warnings)


def audit_upstream(repo):
    """Check supported upstream contract without importing CUDA or running train.py."""
    repo = Path(repo).resolve()
    text = (repo/'scene'/'dataset_readers.py').read_text(encoding="utf-8")
    tree = ast.parse(text)
    reader = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'readColmapCameras')
    matches = [k.value for x in ast.walk(reader) if isinstance(x, ast.Call)
               for k in x.keywords if k.arg == 'time']
    require(any(ast.unparse(x) == 'float(idx / len(cam_extrinsics))' for x in matches),
            'Unsupported upstream COLMAP time assignment; review before training')
    args = ast.parse((repo/'arguments'/'__init__.py').read_text(encoding="utf-8"))
    model = next(x for x in args.body if isinstance(x, ast.ClassDef) and x.name == 'ModelParams')
    defaults = {}
    for x in ast.walk(model):
        if isinstance(x, ast.Assign):
            for target in x.targets:
                if isinstance(target, ast.Attribute):
                    try:
                        defaults[target.attr] = ast.literal_eval(x.value)
                    except (ValueError, TypeError):
                        pass
    require(defaults.get('_white_background') is True and defaults.get('eval') is True,
            'Upstream background/split defaults changed; review training config')
    return dict(path=str(repo), revision=revision(repo), colmap_time='index/N',
                defaults={k: defaults.get(k) for k in ['_white_background','eval','render_process','_resolution']})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', required=True)
    ap.add_argument('--calib')
    ap.add_argument('--fourdgs')
    ap.add_argument('--out', help='optional audit JSON; otherwise only prints')
    args = ap.parse_args()
    report = audit(args.work, args.calib)
    if args.fourdgs:
        report['upstream'] = audit_upstream(args.fourdgs)
    output = json.dumps(report, indent=2)
    print(output)
    if args.out:
        Path(args.out).write_text(output+'\n')


if __name__ == '__main__':
    main()
