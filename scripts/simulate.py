#!/usr/bin/env python
"""Run TIDES on a prepared + trained sequence.

  python scripts/simulate.py --work work/eds/00_peanuts_dark --fourdgs third_party/4DGaussians \
      --config configs/tides_adaptive.yaml --name tides_adaptive
Writes <work>/events/<name>.h5 and <name>.json (config + run statistics).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from tides import events as E  # noqa: E402
from tides.config_io import load_yaml  # noqa: E402
from tides.simulator import TidesConfig, TidesSimulator  # noqa: E402
from tides.workdir import build_renderer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--fourdgs", required=True, help="path to a hustvl/4DGaussians checkout")
    ap.add_argument("--fourdgs-config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "4dgs", "tides_default.py"))
    ap.add_argument("--model", default=None, help="4DGS output folder (default <work>/4dgs)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "tides_10x.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. comparator.n_max=1")
    ap.add_argument("--name", default=None)
    ap.add_argument("--t-start", type=float, default=None)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--tangent-mode", default="forward", choices=["forward", "reverse"])
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg_dict = load_yaml(args.config, args.set)
    if args.seed is not None:
        cfg_dict.setdefault("comparator", {})["seed"] = args.seed
        cfg_dict.setdefault("readout", {})["seed"] = args.seed
    cfg = TidesConfig.from_dict(cfg_dict)
    torch.manual_seed(cfg.comparator.seed)

    r, meta = build_renderer(args.work, args.fourdgs, model_path=args.model, config=args.fourdgs_config,
                             mode=args.tangent_mode, device=args.device, chunk=args.chunk)
    t0 = args.t_start if args.t_start is not None else meta["t_range"][0]
    t1 = args.t_end if args.t_end is not None else meta["t_range"][1]
    sim = TidesSimulator(r, cfg, frame_interval=meta["frame_interval"])
    ev = sim.run(t0, t1)

    name = args.name or os.path.splitext(os.path.basename(args.config))[0]
    out = os.path.join(args.work, "events", name + ".h5")
    E.save(out, ev)
    info = dict(config=cfg.to_dict(), stats=sim.stats, t_range=[t0, t1], tangent_mode=args.tangent_mode)
    with open(os.path.join(args.work, "events", name + ".json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"[simulate] {len(ev['t']):,} events -> {out}")
    print(json.dumps(sim.stats, indent=1))


if __name__ == "__main__":
    main()
