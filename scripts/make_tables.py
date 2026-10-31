#!/usr/bin/env python
"""Aggregate per-sequence metrics into Tables 1, 2 and 4, next to the paper's numbers.

  python scripts/make_tables.py --work-root work --out results/tables.md
Averages over all sequences found under <work-root>/<dataset>/*/metrics.json,
as the paper does ("average across all sequences in each dataset").
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tides.paper_numbers import DATASETS, LABELS, METHODS, TABLE1, TABLE2, TABLE4  # noqa: E402


def collect(work_root):
    agg = {}  # dataset -> method -> metric -> list
    for path in glob.glob(os.path.join(work_root, "*", "*", "metrics.json")):
        r = json.load(open(path))
        ds = r.get("dataset")
        for m, vals in r.get("methods", {}).items():
            for k, v in vals.items():
                if isinstance(v, (int, float)) and v == v:
                    agg.setdefault(ds, {}).setdefault(m, {}).setdefault(k, []).append(v)
    return agg


def mean(agg, ds, m, k):
    v = agg.get(ds, {}).get(m, {}).get(k)
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def fmt(x):
    return "-" if x is None else f"{x:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", default="work")
    ap.add_argument("--out", default="results/tables.md")
    args = ap.parse_args()
    agg = collect(args.work_root)
    lines = ["# Reproduced tables (ours / paper)", "",
             "Each cell is `ours / paper`. Ours is the mean over the sequences we ran (n in brackets).", ""]

    lines += ["## Table 1: event fidelity (IG-NLL, Chamfer; lower is better)", ""]
    hdr = "| Simulator | " + " | ".join(f"{LABELS[d]} IG-NLL | {LABELS[d]} Chamfer" for d in DATASETS) + " |"
    lines += [hdr, "|" + "---|" * (1 + 2 * len(DATASETS))]
    for m in METHODS:
        row = [LABELS[m]]
        for d in DATASETS:
            for j, k in enumerate(("ig_nll", "chamfer")):
                ours, n = mean(agg, d, m, k)
                row.append(f"{fmt(ours)} / {TABLE1[m][d][j]:.6f}" + (f" [{n}]" if n else ""))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Table 2: batching diagnostics (Same-ts, Fano error, ISI spike; lower is better)", ""]
    hdr = "| Simulator | " + " | ".join(f"{LABELS[d]} Same-ts | {LABELS[d]} Fano | {LABELS[d]} ISI spike" for d in DATASETS) + " |"
    lines += [hdr, "|" + "---|" * (1 + 3 * len(DATASETS))]
    for m in METHODS:
        row = [LABELS[m]]
        for d in DATASETS:
            for j, k in enumerate(("same_ts", "fano", "isi_spike")):
                ours, n = mean(agg, d, m, k)
                row.append(f"{fmt(ours)} / {TABLE2[m][d][j]:.6f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Table 4: ablations on EDS (IG-NLL, Chamfer)", "", "| Variant | IG-NLL ours / paper | Chamfer ours / paper |", "|---|---|---|"]
    for key, (label, ig, ch) in TABLE4.items():
        m = "tides_adaptive" if key == "full" else f"ablation_{key}"
        o_ig, n = mean(agg, "eds", m, "ig_nll")
        o_ch, _ = mean(agg, "eds", m, "chamfer")
        lines.append(f"| {label} | {fmt(o_ig)} / {ig:.5f} | {fmt(o_ch)} / {ch:.5f} |")

    lines += ["", "## Ranking check", ""]
    for d in DATASETS:
        for k, j in (("ig_nll", 0), ("chamfer", 1)):
            ours = {m: mean(agg, d, m, k)[0] for m in METHODS[:5]}
            if any(v is None for v in ours.values()):
                continue
            o_rank = sorted(ours, key=ours.get)
            p_rank = sorted(METHODS[:5], key=lambda m: TABLE1[m][d][j])
            lines.append(f"- {LABELS[d]} {k}: ours {' < '.join(LABELS[m] for m in o_rank)}; paper {' < '.join(LABELS[m] for m in p_rank)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
