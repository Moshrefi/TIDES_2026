# TIDES re-implementation

An independent implementation of **TIDES: Time-Derivative Event Simulation via Deformable Reconstruction**
(Thirgood, Ghosh, Hadfield; arXiv 2606.02058v1), written from the paper because no official code is available.

TIDES simulates event-camera data from a dynamic 4D Gaussian Splatting (4DGS) scene. Instead of differencing
rendered frames, it renders the log-luminance **L** and its exact time derivative **L̇** (plus opacity α, α̇ and
two compositing diagnostics) in the same front-to-back order as the image, and solves for threshold-crossing
times in continuous time.

> **Expectations.** This code follows the paper's equations, but the paper leaves its metric definitions,
> sequence lists and about 30 constants to a supplementary file we do not have. Matching rankings,
> trends, or exact numbers has not been validated. Every guessed value is listed in
> [docs/assumptions.md](docs/assumptions.md) and is a config option.

## What is implemented

| Paper | Code |
|---|---|
| Sec. 3.1 EWA splatting, Eq. 1-5 | `tides/projection.py`, `tides/rasterizer.py` (matches the 3DGS CUDA rasterizer's rules) |
| Sec. 3.2 time-JVP of the deformation net, Eq. 6-8 | `tides/renderer.py` (`torch.func.jvp`; exact reverse-mode fallback) |
| Sec. 3.2 derivative compositing, Eq. 9-15 | `tides/rasterizer.py` |
| Camera motion from neighbouring poses | `tides/camera.py` (Hermite position, SLERP rotation) |
| Sec. 3.3 sensor front-end F | `tides/sensor.py` (PSF, centre-surround, photoreceptor low-pass; off by default) |
| Sec. 3.3 residual comparator, Eq. 16-17, multi-crossing | `tides/comparator.py` (linear and quadratic local models) |
| Sec. 3.4 risk score, Eq. 18, adaptive steps, b_min masking | `tides/adaptive.py` |
| Sec. 3.5 tile arbiter, Eq. 19 | `tides/readout.py` |
| Sec. 4.3 metrics (IG-NLL, Chamfer, Same-ts, ISI spike, Fano, Eq. 20) | `tides/metrics/` |
| Sec. 4.1 baselines (ESIM, V2E, ICNS/IEBCS, DVS-Voltmeter) | `tides/baselines/` (wrap the authors' code) |
| Sec. 4.2 datasets (EDS, DSEC, HS-ERGB, BS-ERGB) | `tides/datasets/loaders.py` |
| Tables 1, 2, 4 | `scripts/evaluate.py`, `scripts/make_tables.py`, `scripts/run_ablations.py` |

Table 3 (downstream training of E2VID, TimeLens-XL, Depth Any Event Stream, ESS) is not included yet.

## Install

Tested with Python 3.10-3.13, PyTorch 2.x, one NVIDIA A100 80GB.

```bash
conda create -n tides python=3.10 -y && conda activate tides
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
bash scripts/setup_third_party.sh       # 4DGS + baseline simulators into third_party/
python -m pytest -q tests/              # 19 tests, CPU is enough
python scripts/toy_demo.py              # synthetic end-to-end check, ~2 min on CPU
```

## Running one sequence

```bash
# 0. check the dataset loads (prints frames, calibration, event ranges)
python scripts/inspect_dataset.py --dataset eds --root /data/EDS --seq 00_peanuts_dark

# 1. prepare: undistort frames, poses, initial points, timing metadata
python scripts/prepare_scene.py --dataset eds --root /data/EDS --seq 00_peanuts_dark \
    --out work/eds/00_peanuts_dark --t-start 0 --t-end 5

# 2-4. train 4DGS, simulate TIDES (10x and adaptive), run the 4 baselines, score
bash scripts/run_sequence.sh work/eds/00_peanuts_dark

# Table 4 ablations (EDS)
python scripts/run_ablations.py --work work/eds/00_peanuts_dark --fourdgs third_party/4DGaussians

# Tables 1, 2, 4 next to the paper's numbers
python scripts/make_tables.py --work-root work --out results/tables.md
```

Datasets without ground-truth poses need poses first:

* **DSEC** (static driving scenes): run COLMAP on `images/left/rectified`, then pass `--colmap <sparse/0>`.
* **HS-ERGB, BS-ERGB** (dynamic objects): run Easi3R (or COLMAP with dynamic regions masked), export a COLMAP
  model, then pass `--colmap <sparse/0>`.

Work directories must be laid out as `work/<dataset>/<sequence>` (`dataset` in `eds`, `dsec`, `hsergb`, `bsergb`)
for `make_tables.py` to average per dataset.

## Notes on fidelity to the paper

* The rasterizer implements 3DGS-style rules (16x16 tiles, 3-sigma rectangles, per-tile depth
  order, alpha cutoff/clamping, early stop). `tests/test_core.py` compares against a repository
  brute-force reference; this does not establish numerical parity with upstream CUDA renders.
* Derivative tests compare L-dot and alpha-dot against central finite differences of repository renders.
* The renderer uses a pure PyTorch tile rasterizer. Upstream GPU render parity and paper-level
  fidelity remain unvalidated; use `--chunk` to trade memory for speed.
* Seeds: comparator jitter and arbiter drops are seeded (`--seed`). Run 3 seeds and report mean ± std.

See [docs/assumptions.md](docs/assumptions.md) for every value the paper does not specify.

## Original EDS notebook audit

Use [the audited EDS guide](docs/eds_reproduction_audit.md) for the original archive,
explicit calibration, a read-only geometry/time audit, and safe notebook stage controls.
Regenerate the notebook with `python scripts/build_eds_notebook.py`; verify it with
`python scripts/build_eds_notebook.py --check`. All notebook execution stages default to off.
