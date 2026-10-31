"""YAML config loading with dotted command-line overrides (``comparator.n_max=1``)."""
from __future__ import annotations

import copy

import yaml


def _set(d: dict, dotted: str, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_yaml(path: str | None, overrides=()) -> dict:
    d = {}
    if path:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        base = d.pop("inherit", None)
        if base:
            import os

            parent = load_yaml(os.path.join(os.path.dirname(path), base))
            d = merge(parent, d)
    d = copy.deepcopy(d)
    for o in overrides or ():
        k, v = o.split("=", 1)
        _set(d, k, yaml.safe_load(v))
    return d


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
