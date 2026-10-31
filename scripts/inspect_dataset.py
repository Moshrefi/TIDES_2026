#!/usr/bin/env python
"""Sanity-check that a dataset sequence loads: frames, timestamps, calibration, events.

  python scripts/inspect_dataset.py --dataset eds --root /data/EDS --seq 00_peanuts_dark
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tides import events as E  # noqa: E402
from tides.datasets import load_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--png", default=None, help="write an event frame for visual alignment checks")
    args = ap.parse_args()
    s = load_sequence(args.dataset, args.root, args.seq)
    print(f"{s.dataset}/{s.name}: {len(s.frame_paths)} frames, {s.frame_times[0]:.3f}..{s.frame_times[-1]:.3f}s, "
          f"median dt {np.median(np.diff(s.frame_times)) * 1e3:.2f} ms")
    print("rgb calib:", s.rgb)
    print("event calib:", s.event)
    print("T_ev_rgb:\n", np.round(s.T_ev_rgb, 4))
    print("gt poses:", None if s.gt_poses is None else f"{len(s.gt_poses[0])} poses, {s.gt_poses[0][0]:.3f}..{s.gt_poses[0][-1]:.3f}s")
    t0 = float(s.frame_times[len(s.frame_times) // 2])
    ev = s.load_events(t0, t0 + args.seconds)
    print(f"events in [{t0:.3f}, {t0 + args.seconds:.3f}]s: {len(ev['t']):,}  "
          f"x {ev['x'].min() if len(ev['x']) else '-'}..{ev['x'].max() if len(ev['x']) else '-'}  "
          f"y {ev['y'].min() if len(ev['y']) else '-'}..{ev['y'].max() if len(ev['y']) else '-'}  "
          f"ON fraction {np.mean(ev['p'] > 0) if len(ev['p']) else 0:.2f}  shape {s.event_shape}")
    if args.png and len(ev["t"]):
        from PIL import Image

        Image.fromarray(E.event_frame(ev, s.event_shape, t0, t0 + 0.03)).save(args.png)
        print("wrote", args.png)


if __name__ == "__main__":
    main()
