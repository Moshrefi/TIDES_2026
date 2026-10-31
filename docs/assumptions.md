# Assumptions and unstated details

The paper (arXiv 2606.02058v1) refers to a supplementary file for projection Jacobians, metric definitions and
per-sequence results. We did not have it. Everything below is our choice; each is a config value so it can be
corrected once the supplement is available.

## Simulator

| Item | Paper | Our default | Where |
|---|---|---|---|
| τ_max | "< 1" | 0.99 (3DGS) | `rasterizer.py` |
| α cut-off, early stop | not stated | 1/255, T < 1e-4 (3DGS) | `rasterizer.py` |
| ε in log(Y + ε) | not stated | 1e-3 (Y in [0, 1]) | `renderer.py` |
| Luminance weights w | not stated | Rec. 709 | `renderer.py` |
| Coverage proxy b | "accumulated blending mass" | Σ_k w_k (= 1 − T_K) | `rasterizer.py` |
| C+, C− | not stated | 0.2 / 0.2 | `configs/tides_10x.yaml` |
| Threshold jitter, leak | optional | off | config |
| N_max | not stated | 8 | config |
| Local model | linear, optional quadratic | quadratic (S1 is always rendered in our stepping) | config |
| Window starting past a threshold | not stated | emit catch-up events at window start | `comparator.py` |
| Masked pixels (b < b_min) | "suppress emission" | no events, reference re-anchored to S1 | `comparator.py` |
| b_min | not stated | 0.05 | config |
| Risk weights w_b, w_c, w_α | not stated | 1/3 each | config |
| norm(·) | not stated | b as is; c / 32; |α̇| × base step; clipped to [0, 1] | `adaptive.py` |
| κ(r) | in (0, 1] | 1 / (1 + 7r) | `adaptive.py` |
| Step aggregation | not stated | global step from the 99th-percentile risk of emitting pixels | `adaptive.py` |
| Base step (adaptive) | not stated | frame interval / 4, floor frame interval / 64 | config |
| Rate controller | implied by Table 4 ("no controller" vs "no risk") | ≤ 4 predicted crossings per step at the 99th-percentile pixel | `adaptive.py` |
| Tile size, scanner tick | not stated | 32 px, 1 µs | config |
| K0, p0, σ0, β_K, η, γ_σ | not stated | 4, 0, 1 µs, 0.5, 0.05, 10 µs | config |
| ψ_m | "aggregates" three cues | mean of the three cues scaled to [0, 1] | `readout.py` |
| Camera rates | "from neighbouring timestamps" | Hermite spline with central-difference tangents; SLERP rotation | `camera.py` |
| Background colour | not stated | white, matching 4DGS training default | `prepare_scene.py` |

## 4DGS scene fitting

* Upstream 4DGS (hustvl/4DGaussians) with its HyperNeRF config (`configs/4dgs/tides_default.py`), 14k iterations.
* 4DGS assumes a centred principal point, so `prepare_scene.py` undistorts and recentres the RGB frames. The
  simulator renders straight into the (undistorted) event camera, using the dataset's RGB→event extrinsics.
* Frame *i* of *N* gets scene time i/N (what 4DGS's COLMAP reader does); seconds map to scene time piecewise
  linearly.
* EDS ground-truth poses are read as RGB-camera poses (`--gt-pose-frame event` switches this).
* Without a COLMAP point cloud, points are initialised randomly inside the camera frusta.

## Metrics

| Metric | Paper | Ours |
|---|---|---|
| IG-NLL | NLL of real ISIs under an IG fitted to simulated ISIs | per-pixel ISIs in ms; reported as the *excess* NLL over an IG fitted to the real ISIs (the paper's values ~1e-3 suggest some normalisation). `variant: raw` gives the plain NLL. |
| Chamfer | (x, y, v t), polarity-matched, fixed windows, voxel de-dup | 50 ms windows, 1×1×1 voxels, distance divided by the image diagonal, v = opacity-weighted median screen speed of the scene |
| Same-ts | fraction sharing a timestamp within a window | 1 µs resolution, 50 ms windows |
| ISI spike | ISIs near zero or at clock quanta | ISI < 5 µs plus histogram bins > 5× local median (1 µs bins up to 50 ms) |
| Fano error | Eq. 20 over fixed spatiotemporal bins | 8×8 px tiles, Fano over 10 ms counts per tile |

## Baselines

All baselines get the shared renders at frame interval / 10. Thresholds are 0.2 for all. V2E uses its
command-line defaults, DVS-Voltmeter its DVS346 parameters, IEBCS the values from its video example with
radiance scaled to 10 klux. ESIM uses `esim_torch` when it builds, otherwise an equivalent re-implementation.

## Known paper inconsistencies

* Table 1: DVS-Voltmeter's HS-ERGB Chamfer (0.014871) is slightly better than TIDES 10x (0.014892), although the
  text says TIDES is best everywhere.
* Fig. 3's caption names V2E and ESIM, but the figure shows V2E and DVS-Voltmeter.
