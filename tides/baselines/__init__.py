"""Frame-driven baseline simulators fed with the same 4DGS renders (Sec. 4.1-4.2).

All baselines receive the shared renderer's luminance frames on a fixed schedule
(frame interval x ``fraction``; the paper's fixed 10x budget is fraction = 0.1).
Each adapter wraps the authors' public code:

    esim           rpg_vid2e  esim_torch  (pip install rpg_vid2e/esim_torch)
    v2e            SensorsINI/v2e         (pip install v2e)
    dvs_voltmeter  Lynn0306/DVS-Voltmeter (clone; pass its path)
    iebcs          neuromorphicsystems/IEBCS (clone; pass its path)

If esim_torch is not installed, ESIM falls back to an equivalent
re-implementation (per-pixel linear interpolation of log intensity between
frames, all crossings emitted), flagged in the output metadata.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from .. import events as E


class Baseline:
    name = "base"

    def feed(self, Y: np.ndarray, t: float):
        """Y: (H, W) linear luminance in [0, 1]; t: seconds (relative)."""
        raise NotImplementedError

    def result(self) -> dict:
        raise NotImplementedError

    meta: dict = {}


# ----------------------------------------------------------------------------
class ESIM(Baseline):
    name = "esim"

    def __init__(self, shape, c_pos=0.2, c_neg=0.2, refractory_s=0.0, log_eps=1e-3, device="cuda"):
        self.eps = log_eps
        self.chunks = []
        try:
            import esim_torch

            self.impl = esim_torch.ESIM(c_neg, c_pos, int(refractory_s * 1e9))
            self.official = True
        except Exception:
            from ..comparator import Comparator, ComparatorConfig

            self.impl = Comparator(ComparatorConfig(c_pos=c_pos, c_neg=c_neg, n_max=10_000, model="linear"), shape, device=device)
            self.official = False
        self.device = device
        self.prev = None
        self.meta = {"official_code": self.official}

    def feed(self, Y, t):
        L = torch.log(torch.as_tensor(Y, dtype=torch.float32, device=self.device) + self.eps)
        if self.official:
            ev = self.impl.forward(L, torch.tensor(int(round(t * 1e9)), dtype=torch.int64, device=self.device))
            if ev is not None and len(ev["t"]):
                self.chunks.append(dict(t=ev["t"].cpu().numpy().astype(np.float64) * 1e-9, x=ev["x"].cpu().numpy(),
                                        y=ev["y"].cpu().numpy(), p=np.where(ev["p"].cpu().numpy() > 0, 1, -1).astype(np.int8)))
            return
        if self.prev is None:
            self.impl.reset(L)
        else:
            L0, t0 = self.prev
            ev = self.impl.step(t0, t - t0, L0, (L - L0) / (t - t0))
            if len(ev["t"]):
                self.chunks.append({k: v.cpu().numpy() for k, v in ev.items()})
        self.prev = (L, t)

    def result(self):
        return E.sort(E.concat(self.chunks))


# ----------------------------------------------------------------------------
class V2E(Baseline):
    name = "v2e"

    def __init__(self, shape, device="cuda", **params):
        from v2ecore.emulator import EventEmulator

        # v2e command-line defaults
        defaults = dict(pos_thres=0.2, neg_thres=0.2, sigma_thres=0.03, cutoff_hz=300.0, leak_rate_hz=0.01,
                        shot_noise_rate_hz=0.001, refractory_period_s=0.0)
        defaults.update(params)
        H, W = shape
        self.emu = EventEmulator(output_width=W, output_height=H, device=device, **defaults)
        self.chunks = []
        self.meta = {"params": defaults}

    def feed(self, Y, t):
        ev = self.emu.generate_events(np.clip(Y, 0, 1) * 255.0, t)
        if ev is not None and len(ev):
            ev = np.asarray(ev)
            self.chunks.append(dict(t=ev[:, 0].astype(np.float64), x=ev[:, 1].astype(np.int32), y=ev[:, 2].astype(np.int32),
                                    p=np.where(ev[:, 3] > 0, 1, -1).astype(np.int8)))

    def result(self):
        return E.sort(E.concat(self.chunks))


# ----------------------------------------------------------------------------
def _import_from(path, module):
    if path not in sys.path:
        sys.path.insert(0, path)
    return __import__(module, fromlist=["*"])


class DVSVoltmeter(Baseline):
    name = "dvs_voltmeter"

    def __init__(self, shape, repo_path, camera_type="DVS346", K=None):
        cfgmod = _import_from(repo_path, "src.config")
        simmod = _import_from(repo_path, "src.simulator")
        cfg = cfgmod.cfg
        cfg.SENSOR.CAMERA_TYPE = camera_type
        if K is not None:
            cfg.SENSOR.K = list(K)
        elif camera_type == "DVS240":
            cfg.SENSOR.K = [0.000094 * 47065, 23, 0.0002, 1e-7, 5e-8, 0.00001]
        else:
            cfg.SENSOR.K = [0.00018 * 29250, 20, 0.0001, 1e-7, 5e-9, 0.00001]
        self.sim = simmod.EventSim(cfg=cfg, output_folder="/tmp", video_name="tides")
        self.chunks = []
        self.meta = {"camera_type": camera_type, "K": list(cfg.SENSOR.K)}

    def feed(self, Y, t):
        ev = self.sim.generate_events((np.clip(Y, 0, 1) * 255.0).astype(np.float64), int(round(t * 1e6)))
        if ev is not None and len(ev):
            ev = np.asarray(ev)
            self.chunks.append(dict(t=ev[:, 0].astype(np.float64) * 1e-6, x=ev[:, 1].astype(np.int32), y=ev[:, 2].astype(np.int32),
                                    p=np.where(ev[:, 3] > 0, 1, -1).astype(np.int8)))

    def result(self):
        return E.sort(E.concat(self.chunks))


# ----------------------------------------------------------------------------
class IEBCS(Baseline):
    name = "iebcs"

    def __init__(self, shape, repo_path, c_pos=0.2, c_neg=0.2, th_noise=0.01, lat=100, jit=10, ref=100, tau=40,
                 bgnp=0.1, bgnn=0.01, lux=1e4, seed=0):
        mod = _import_from(os.path.join(repo_path, "src"), "dvs_sensor")
        np.random.seed(seed)
        H, W = shape
        self.dvs = mod.DvsSensor("tides")
        self.dvs.initCamera(W, H, lat=lat, jit=jit, ref=ref, tau=tau, th_pos=c_pos, th_neg=c_neg,
                            th_noise=th_noise, bgnp=bgnp, bgnn=bgnn)
        self.lux = lux
        self.t_prev = None
        self.t0 = None
        self.chunks = []
        self.meta = dict(lat=lat, jit=jit, ref=ref, tau=tau, th_noise=th_noise, bgnp=bgnp, bgnn=bgnn, lux=lux)

    def feed(self, Y, t):
        img = np.clip(Y, 0, None).astype(np.float64) * self.lux
        if self.t_prev is None:
            self.dvs.init_image(img)
            self.t_prev = self.t0 = t
            return
        dt_us = int(round((t - self.t_prev) * 1e6))
        buf = self.dvs.update(img, dt_us)
        self.t_prev = t
        n = buf.i
        if n:
            self.chunks.append(dict(t=self.t0 + buf.ts[:n].astype(np.float64) * 1e-6, x=buf.x[:n].astype(np.int32),
                                    y=buf.y[:n].astype(np.int32), p=np.where(buf.p[:n] > 0, 1, -1).astype(np.int8)))

    def result(self):
        return E.sort(E.concat(self.chunks))


def make_baseline(name, shape, third_party: str = "third_party", device="cuda", **kw):
    if name == "esim":
        return ESIM(shape, device=device, **kw)
    if name == "v2e":
        return V2E(shape, device=device, **kw)
    if name == "dvs_voltmeter":
        return DVSVoltmeter(shape, os.path.join(third_party, "DVS-Voltmeter"), **kw)
    if name == "iebcs":
        return IEBCS(shape, os.path.join(third_party, "IEBCS"), **kw)
    raise ValueError(name)


BASELINES = ("esim", "v2e", "iebcs", "dvs_voltmeter")
