#!/usr/bin/env python
"""Score simulated event streams against the real events (Tables 1, 2, 4 metrics).

  python scripts/evaluate.py --work work/eds/00_peanuts_dark --methods esim v2e iebcs dvs_voltmeter tides_10x tides_adaptive
Writes <work>/metrics.json (merged with any previous results).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tides import events as E  # noqa: E402
from tides.config_io import load_yaml  # noqa: E402
from tides.datasets import load_sequence  # noqa: E402
from tides.metrics import chamfer_st, fano_error, ig_nll, isi_spike_ratio, same_ts_fraction  # noqa: E402
from tides.workdir import load_meta  # noqa: E402


def estimate_speed(work, fourdgs, fourdgs_config, device, n=8, tangent_mode="reverse"):
    """Median screen-space speed (px/s) of visible Gaussians: the Chamfer 'v'."""
    import torch

    from tides.workdir import build_renderer

    r, meta = build_renderer(work, fourdgs, config=fourdgs_config, device=device, mode=tangent_mode)
    t0, t1 = meta["t_range"]
    speeds = []
    for t in np.linspace(t0, t1, n + 2)[1:-1]:
        proj, tang = r.primitives(float(t))
        u = proj["means2d"]
        vis = (proj["radius"] > 0) & (u[:, 0] >= 0) & (u[:, 0] < r.K.width) & (u[:, 1] >= 0) & (u[:, 1] < r.K.height)
        w = proj["opacity"][vis]
        sp = tang["means2d"][vis].norm(dim=-1)
        if len(sp):
            # opacity-weighted median
            o = torch.argsort(sp)
            cw = torch.cumsum(w[o], 0)
            speeds.append(float(sp[o][torch.searchsorted(cw, cw[-1] / 2)]))
    return float(np.median(speeds)) if speeds else 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "metrics.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--fourdgs", default=None, help="needed only for chamfer.v = auto")
    ap.add_argument("--fourdgs-config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "4dgs", "tides_default.py"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tangent-mode", choices=["forward", "reverse"], default="reverse",
                    help="AD mode for automatic Chamfer speed; reverse supports upstream grid_sample")
    ap.add_argument("--calib", default=None, help="explicit EDS calibration, also supports legacy metadata")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config, args.set)
    meta = load_meta(args.work)
    kw = dict(meta.get("loader_kwargs", {}))
    if args.calib:
        if meta["dataset"] != "eds":
            ap.error("--calib is supported only for EDS")
        kw["calib_path"] = args.calib
    seq = load_sequence(meta["dataset"], meta["root"], meta["sequence"], **kw)
    if meta["dataset"] == "eds":
        from audit_prepared_scene import audit
        audit(args.work, seq.calibration_path)
    prep_hash = hashlib.sha256(open(os.path.join(args.work, "tides_meta.json"), "rb").read()).hexdigest()
    K = meta["K_render"]
    shape = (K["height"], K["width"])

    out_path = args.out or os.path.join(args.work, "metrics.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}

    if results.get("methods") and (results.get("metric_config") != cfg
                                   or results.get("preparation_sha256") != prep_hash):
        raise ValueError("Existing metrics use different or unrecorded configuration/preparation. "
                         "Choose a new --out; do not merge incompatible runs.")
    v = cfg["chamfer"].get("v", "auto")
    if v == "auto":
        if "speed_px_s" not in results:
            assert args.fourdgs, "chamfer.v=auto needs --fourdgs (or set chamfer.v=<px/s>)"
            results["speed_px_s"] = estimate_speed(args.work, args.fourdgs, args.fourdgs_config, args.device, tangent_mode=args.tangent_mode)
        v = results["speed_px_s"]
    print(f"[evaluate] chamfer v = {v:.1f} px/s")

    real_cache = {}
    for m in args.methods:
        path = os.path.join(args.work, "events", m + ".h5")
        if not os.path.exists(path):
            print(f"[evaluate] missing {path}, skipped")
            continue
        sim = E.load(path)
        info_path = os.path.join(args.work, "events", m + ".json")
        if not os.path.exists(info_path):
            raise ValueError(f"Missing {info_path}; explicit simulation interval is required, including for empty streams")
        t0, t1 = json.load(open(info_path))["t_range"]
        if not (meta["t_range"][0] <= t0 < t1 <= meta["t_range"][1]):
            raise ValueError(f"{m}: event interval outside prepared scene")
        key = (round(t0, 6), round(t1, 6))
        if key not in real_cache:
            real_cache[key] = seq.load_events(t0, t1)
        real = real_cache[key]
        sim = E.slice_time(sim, t0, t1)
        W = shape[1]
        c = cfg
        res = dict(
            t_range=[t0, t1],
            n_sim=int(len(sim["t"])), n_real=int(len(real["t"])),
            ig_nll=ig_nll(real, sim, W, unit=c["ig_nll"]["unit"], variant=c["ig_nll"]["variant"], by_polarity=c["ig_nll"]["by_polarity"]),
            chamfer=chamfer_st(real, sim, v, shape, window=c["chamfer"]["window"], voxel=c["chamfer"]["voxel"], normalize=c["chamfer"]["normalize"]),
            same_ts=same_ts_fraction(sim, **c["same_ts"]),
            fano=fano_error(real, sim, shape, **c["fano"]),
            isi_spike=isi_spike_ratio(sim, W, **c["isi_spike"]),
        )
        results.setdefault("methods", {})[m] = res
        print(f"[evaluate] {m}: " + "  ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))
    # real-data reference values for the batching diagnostics
    if real_cache:
        real = next(iter(real_cache.values()))
        results["real"] = dict(same_ts=same_ts_fraction(real, **cfg["same_ts"]), isi_spike=isi_spike_ratio(real, shape[1], **cfg["isi_spike"]),
                               n=int(len(real["t"])))
    results["dataset"], results["sequence"] = meta["dataset"], meta["sequence"]
    results["metric_config"] = cfg
    results["speed_tangent_mode"] = args.tangent_mode
    results["preparation_sha256"] = prep_hash
    results["calibration_path"] = seq.calibration_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[evaluate] wrote {out_path}")


if __name__ == "__main__":
    main()
