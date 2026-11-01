#!/usr/bin/env python
"""Run the frame-driven baselines on the same 4DGS renders (Sec. 4.2 protocol).

One pass renders the luminance frames at frame_interval * fraction and feeds
every requested baseline, so they all see identical inputs.

  python scripts/run_baselines.py --work work/eds/00_peanuts_dark --fourdgs third_party/4DGaussians \
      --third-party third_party --methods esim v2e iebcs dvs_voltmeter
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tides import events as E  # noqa: E402
from tides.baselines import BASELINES, make_baseline  # noqa: E402
from tides.config_io import load_yaml  # noqa: E402
from tides.workdir import build_renderer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--fourdgs", required=True)
    ap.add_argument("--fourdgs-config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "4dgs", "tides_default.py"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--third-party", default="third_party")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "baselines.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--methods", nargs="+", default=list(BASELINES))
    ap.add_argument("--t-start", type=float, default=None)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-frames", action="store_true", help="also store the rendered frames (large)")
    args = ap.parse_args()

    cfg = load_yaml(args.config, args.set)
    r, meta = build_renderer(args.work, args.fourdgs, model_path=args.model, config=args.fourdgs_config, device=args.device)
    shape = (r.K.height, r.K.width)
    sims = {}
    for m in args.methods:
        kw = dict(cfg.get(m, {}))
        if m == "esim":
            kw["log_eps"] = cfg.get("log_eps", 1e-3)
        sims[m] = make_baseline(m, shape, third_party=args.third_party, device=args.device, **kw)

    t0 = args.t_start if args.t_start is not None else meta["t_range"][0]
    t1 = args.t_end if args.t_end is not None else meta["t_range"][1]
    dt = meta["frame_interval"] * cfg.get("fraction", 0.1)
    if not (meta["t_range"][0] <= t0 < t1 <= meta["t_range"][1]):
        raise ValueError("Requested interval is outside the prepared scene")
    times = np.arange(t0, t1, dt)
    # Feed the endpoint too, so every baseline covers the declared scoring window.
    times = np.append(times, t1)
    frames = []
    tic = time.time()
    for i, t in enumerate(times):
        Y = r.render(float(t), with_tangent=False).Y.float().cpu().numpy()
        for s in sims.values():
            s.feed(Y, float(t))
        if args.save_frames:
            frames.append(Y.astype(np.float16))
        if i % 100 == 0:
            print(f"[baselines] frame {i}/{len(times)}  {time.time() - tic:.0f}s", flush=True)
    os.makedirs(os.path.join(args.work, "events"), exist_ok=True)
    for m, s in sims.items():
        ev = s.result()
        E.save(os.path.join(args.work, "events", m + ".h5"), ev)
        with open(os.path.join(args.work, "events", m + ".json"), "w") as f:
            json.dump(dict(meta=s.meta, n_frames=len(times), frame_dt=dt, t_range=[t0, t1]), f, indent=1)
        print(f"[baselines] {m}: {len(ev['t']):,} events")
    if args.save_frames:
        np.savez_compressed(os.path.join(args.work, "renders_baseline.npz"), Y=np.stack(frames), t=times)


if __name__ == "__main__":
    main()
