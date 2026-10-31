"""TIDES event simulator: ties renderer, comparator, adaptive stepping and readout together."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from . import events as E
from .adaptive import AdaptiveConfig, StepController
from .comparator import Comparator, ComparatorConfig
from .readout import ReadoutConfig, TileArbiter, tile_activity
from .sensor import SensorConfig, SensorFrontEnd


@dataclass
class TidesConfig:
    comparator: ComparatorConfig = field(default_factory=ComparatorConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    # "analytic" = time-tangent renderer (TIDES). "finite_difference" = Table 4 ablation:
    # slope from differencing consecutive renders, linear model.
    slope: str = "analytic"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None):
        d = d or {}
        return cls(
            comparator=ComparatorConfig(**d.get("comparator", {})),
            sensor=SensorConfig(**d.get("sensor", {})),
            adaptive=AdaptiveConfig(**d.get("adaptive", {})),
            readout=ReadoutConfig(**d.get("readout", {})),
            slope=d.get("slope", "analytic"),
        )


class TidesSimulator:
    def __init__(self, renderer, cfg: TidesConfig, frame_interval: float):
        self.r = renderer
        self.cfg = cfg
        H, W = renderer.K.height, renderer.K.width
        self.shape = (H, W)
        cfg.adaptive.contrast = 0.5 * (cfg.comparator.c_pos + cfg.comparator.c_neg)
        self.ctrl = StepController(cfg.adaptive, frame_interval)
        self.sensor = SensorFrontEnd(cfg.sensor)
        self.comp = Comparator(cfg.comparator, self.shape, device=renderer.device)
        self.arbiter = TileArbiter(cfg.readout, self.shape)
        self.stats = {}

    def _signal(self, frame):
        return self.sensor(frame.t, frame.L, frame.L_dot)

    def run(self, t_start: float, t_end: float, progress: bool = True, max_steps: int | None = None):
        cfg = self.cfg
        analytic = cfg.slope == "analytic"
        if not analytic:
            # Table 4 "Finite-diff slope": the slope comes from differencing
            # consecutive renders (linear model). Tangents are still rendered so
            # the risk / readout cues stay identical.
            self.comp.cfg.model = "linear"
        tic = time.time()
        f0 = self.r.render(t_start, with_tangent=True)
        S0, Sd0 = self._signal(f0)
        self.comp.reset(S0)
        chunks, t, steps, dts = [], float(t_start), 0, []
        contrast = 0.5 * (cfg.comparator.c_pos + cfg.comparator.c_neg)
        last_print = tic
        while t < t_end - 1e-12:
            dt = min(self.ctrl.next_dt(f0), t_end - t)
            if dt <= 0:
                break
            f1 = self.r.render(t + dt, with_tangent=True)
            S1, Sd1 = self._signal(f1)
            slope = Sd0 if analytic else (S1 - S0) / dt
            mask = self.ctrl.emit_mask(f0)
            ev = self.comp.step(t, dt, S0, slope, S1, emit_mask=mask)
            if cfg.readout.enabled:
                self.arbiter.record(t, t + dt, tile_activity(f0, contrast, self.ctrl.dt_base, cfg.readout.tile, cfg.readout.c_ref,
                                                                       use_alpha=cfg.adaptive.use_alpha_cue))
            if len(ev["t"]):
                chunks.append({k: v.cpu().numpy() for k, v in ev.items()})
            f0, S0, Sd0 = f1, S1, Sd1
            t += dt
            steps += 1
            dts.append(dt)
            if progress and time.time() - last_print > 10:
                last_print = time.time()
                n = sum(len(c["t"]) for c in chunks)
                print(f"[tides] t={t - t_start:.4f}/{t_end - t_start:.4f}s steps={steps} events={n:,}", flush=True)
            if max_steps and steps >= max_steps:
                break
        ev = E.sort(E.concat(chunks))
        ev["x"] = ev["x"].astype(np.int32)
        ev["y"] = ev["y"].astype(np.int32)
        ideal_n = len(ev["t"])
        if cfg.readout.enabled:
            ev = self.arbiter(ev)
        self.stats = dict(
            steps=steps,
            render_queries=steps + 1,
            mean_dt=float(np.mean(dts)) if dts else 0.0,
            min_dt=float(np.min(dts)) if dts else 0.0,
            events_ideal=ideal_n,
            events_out=len(ev["t"]),
            seconds=time.time() - tic,
        )
        return ev
