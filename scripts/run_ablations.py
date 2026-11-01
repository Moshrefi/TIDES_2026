#!/usr/bin/env python
"""Run the Table 4 ablations on one prepared EDS sequence, then score them.

  python scripts/run_ablations.py --work work/eds/00_peanuts_dark --fourdgs third_party/4DGaussians
Each variant is configs/tides_adaptive.yaml plus the overrides in configs/ablations.yaml.
The compute-matched variant uses the fixed step that gives the same number of
renderer queries as the full adaptive run.
"""
import argparse
import json
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--fourdgs", required=True)
    ap.add_argument("--fourdgs-config", default=os.path.join(HERE, "..", "configs", "4dgs", "tides_default.py"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tangent-mode", choices=["forward", "reverse"], default="reverse")
    args = ap.parse_args()
    original_run = run
    def run_with_config(cmd):
        cmd = list(cmd) + ["--fourdgs-config", args.fourdgs_config, "--tangent-mode", args.tangent_mode]
        if cmd[1].endswith("simulate.py"):
            cmd += ["--seed", str(args.seed)]
        original_run(cmd)

    abl = yaml.safe_load(open(os.path.join(HERE, "..", "configs", "ablations.yaml")))
    base = os.path.join(HERE, "..", "configs", "tides_adaptive.yaml")
    names = []
    for key, overrides in abl.items():
        if args.only and key not in args.only:
            continue
        name = "tides_adaptive" if key == "full" else f"ablation_{key}"
        overrides = list(overrides)
        if key == "compute_matched":
            full_info = os.path.join(args.work, "events", "tides_adaptive.json")
            if not os.path.exists(full_info):
                run_with_config([sys.executable, os.path.join(HERE, "simulate.py"), "--work", args.work, "--fourdgs", args.fourdgs,
                     "--config", base, "--name", "tides_adaptive", "--device", args.device])
            info = json.load(open(full_info))
            meta = json.load(open(os.path.join(args.work, "tides_meta.json")))
            steps = info["stats"]["steps"]
            n_frames = (info["t_range"][1] - info["t_range"][0]) / meta["frame_interval"]
            overrides.append(f"adaptive.fixed_fraction={n_frames / max(steps, 1):.8f}")
        if key == "full" and os.path.exists(os.path.join(args.work, "events", "tides_adaptive.h5")):
            names.append(name)
            continue
        run_with_config([sys.executable, os.path.join(HERE, "simulate.py"), "--work", args.work, "--fourdgs", args.fourdgs,
             "--config", base, "--name", name, "--device", args.device, "--set", *overrides] if overrides else
            [sys.executable, os.path.join(HERE, "simulate.py"), "--work", args.work, "--fourdgs", args.fourdgs,
             "--config", base, "--name", name, "--device", args.device])
        names.append(name)
    run_with_config([sys.executable, os.path.join(HERE, "evaluate.py"), "--work", args.work, "--fourdgs", args.fourdgs,
         "--device", args.device, "--methods", *names])


if __name__ == "__main__":
    main()
