# EDS notebook audit and proposed fixes

Status: **approved by user for publication on a dedicated branch; no training run by the agent**.

Base: `Moshrefi/TIDES_2026`, branch `paper-reproduction-notebook`, commit
`0a18e87095aee9d081a3784a2bcc99e15b64f08f`. Local working branch: `fix/eds-reproduction-audit`.
The original branch is unchanged. This document describes the audited changes and remaining limitations.

## Findings and decisions

| Area | Finding | Proposed behavior |
|---|---|---|
| Notebook dataset root | Default pointed at an EvTween preprocessed copy | Default to `/u/marmary/evtween/EDS/EDS`; env overrides require no cell edits |
| Calibration | Original discovery missed `calib/00_calib`; an invalid explicit env path could silently fall back | `--calib` in inspection, preparation, evaluation; explicit missing file fails; ambiguous discovery fails; resolved path and SHA256 recorded |
| Camera assignment | Official YAML identifies cam0 as RGB, cam1 as event; their intrinsics/extrinsic match the user's preview | Preserve this assignment and `T_event_rgb`; check metadata against the supplied YAML |
| Pose direction | TUM c2w is inverted to OpenCV/COLMAP w2c | Preserve `R_wc=R_cw.T`, `t_wc=-R_wc*C`; test with a rotated/translated camera; compare COLMAP poses to metadata |
| GT physical camera | Existing loader assumes RGB; provided logs do not prove the GT file's physical frame | Preserve explicit assumption; expose `TIDES_GT_POSE_FRAME`; do not claim it is verified |
| Image orientation | Official cam0 YAML contains `flip: True`; loader preserves stored images | Warn; verify whether the archive already applied the flip before calibrated training; do not guess a flip axis or double-flip |
| Time mapping | Upstream COLMAP reader uses enumeration index/N before train/test split | Preserve `i/N`, including 112/113 for the 113th frame; irregular real timestamps map piecewise linearly; validate finite/strictly increasing inputs |
| Preparation reruns | Old images and binary COLMAP files could survive re-preparation | Refuse nonempty existing scene output; audit rejects stale images/binary models; never replace user's prepared scene |
| Image sizes | Old preparation silently resized images to calibration dimensions | Fail on mismatch so incorrect calibration isn't hidden |
| Training flags | Upstream booleans use `store_true`; strings such as `--eval False` are invalid | Set config values explicitly: white background, eval=True, llffhold=8, resolution=1, render_process=False; no invented upstream seed flag |
| Dependency paths | Notebook assumed launch from root and bundled checkout | Find root from root/notebooks; allow explicit existing 4DGS/third-party paths; add notebook extras file; never install automatically |
| Checkpoint existence | Directory existence was treated as a trained checkpoint; enabling train could overwrite an output | Notebook requires fine-stage PLY + deformation.pth; train starts when output is absent, reuses a complete checkpoint, and rejects incomplete output |
| Baseline scoring window | Last rendered frame could precede declared interval end | Feed endpoint as well; final spacing can be shorter than the nominal fixed step |
| Evaluation | Missing sidecar inferred interval from event extrema; empty streams could crash; merged old metrics could mix settings | Require declared interval, check scene bounds/calibration, store per-method ranges and metadata fingerprint; incompatible/legacy merge fails; select fresh `--out` |
| Ablations/shell | Custom model config not consistently forwarded | Forward the same 4DGS config; pass simulation seed through ablations; shell audits first and avoids retraining incomplete outputs |
| Paper claims | A short clip and assumption-based metrics cannot establish reproduction | Label clip/available-sequence means as diagnostic; remove ranking guarantee; record source/config/environment provenance |

The 113-frame preparation, 1031 source frames, 2013 GT poses, and 1.493700027 s endpoint
are **user-reported prior results**, not measurements repeated in this audit.
The official YAML has RGB 640x480, fx=766.536025, fy=767.574946,
cx=291.050351, cy=227.406048; recentring/cropping gives **583x455** with cx=291, cy=227.
The event camera remains **640x480**, fx=560.852095, fy=560.629582.
Thus different training/render intrinsics and scene time 112/113 are not themselves defects.

## Geometry convention

For a TUM camera-to-world pose `[R_cw,C]`, the exported pose is
`x_rgb = R_cw.T @ x_world - R_cw.T @ C`.
Kalibr cam1's `T_cn_cnm1` maps cam0/RGB coordinates into cam1/event coordinates:
`T_event_world = T_event_rgb @ T_rgb_world`.
If the GT input is actually event-camera c2w, first use
`T_rgb_world = inverse(T_event_rgb) @ T_event_world`.
Scale affects translations in both poses and the inter-camera baseline, not rotations.
No OpenGL y/z flip is introduced into these OpenCV/COLMAP transforms.

These algebra checks cannot determine which physical camera a TUM file describes.
The supplied GT file header/exporter and raw RGB/event alignment must establish that.
The audit does not decode every image, validate point-cloud quality, or compare a GPU render.

## Reproducible notebook

`notebooks/01_eds_paper_reproduction.ipynb` is generated from
`scripts/build_eds_notebook.py`. Use:

```bash
python scripts/build_eds_notebook.py
python scripts/build_eds_notebook.py --check
```

All execution stages are off by default. Notebook Run All with no stage selection previews
the training command and reports available files. It does not train, simulate, install,
aggregate, or write a manifest automatically.

On the user's Linux machine, after these proposed fixes are approved and made available:

```bash
cd /u/marmary/airlab_v2e/TIDES_2026
export TIDES_EDS_ROOT=/u/marmary/evtween/EDS/EDS
export TIDES_EDS_CALIB=/u/marmary/evtween/EDS/calib/00_calib/camchain-mediajaviJAVISdatasetshwdscalibratione2kalibr.yaml
export TIDES_WORK=/u/marmary/airlab_v2e/TIDES_2026/work/eds/00_peanuts_dark
# Set this to the existing upstream checkout if it is outside third_party:
export TIDES_FOURDGS=/absolute/path/to/4DGaussians

# READ-ONLY audit of the existing successful preparation; no need to prepare again:
/u/marmary/.conda/envs/tides/bin/python scripts/audit_prepared_scene.py \
  --work "$TIDES_WORK" --calib "$TIDES_EDS_CALIB" --fourdgs "$TIDES_FOURDGS"
```

Use the actual checkout path for TIDES_FOURDGS. The audit fails if the upstream COLMAP
reader's time convention changed. For a fresh preparation, select a new work directory:

```bash
/u/marmary/.conda/envs/tides/bin/python scripts/prepare_scene.py \
  --dataset eds --root "$TIDES_EDS_ROOT" --seq 00_peanuts_dark \
  --calib "$TIDES_EDS_CALIB" --gt-pose-frame rgb \
  --out work/eds/00_peanuts_dark_audited --t-start 0 --t-end 1.5 --seed 0
```

`rgb` here records the current assumption; change only after the GT camera is established.
This proposed preparation never overwrites `work/eds/00_peanuts_dark`.

Install notebook extras into the existing `tides` environment if needed, select its kernel,
and launch Jupyter from the repository root or notebooks folder:

```bash
/u/marmary/.conda/envs/tides/bin/python -m pip install -r requirements-notebook.txt
export TIDES_RUN_STAGES=inspect,audit,record
/u/marmary/.conda/envs/tides/bin/python -m jupyterlab notebooks/01_eds_paper_reproduction.ipynb
```

The export is read by the notebook kernel. Restart the kernel / rerun settings after changing it.
`inspect,audit,record` does not train. `record` writes only a reproducibility manifest.
Later stages, when explicitly intended, are named `train`, `tides`, `baselines`, `evaluate`,
`ablations`, `tables`. Start with one stage at a time. Training requires a CUDA environment
with 4DGS's pinned/local dependencies and compiled extensions; a source audit isn't an import/build test.
`TIDES_SEED` controls preparation and TIDES randomness; it does not claim to control
upstream training or all stochastic baseline libraries. A record of the upstream revision is retained.

For legacy metrics, choose a new `TIDES_METRICS` path when the old file lacks provenance.
Ablations and make_tables use standard `metrics.json`; custom output files are not automatically aggregated.

## Checks performed here

- Six CPU-only audit regression tests passed: irregular time, duplicate time, incorrect normalization,
  mismatched pose, stale image/binary model, nonrigid extrinsic.
- Four comparison-report tests passed: signed/percent differences, missing/nonfinite values, full reference-row coverage, and rejecting non-EDS input.
- Two notebook contract tests passed: deterministic regeneration and default Run All from root/notebooks
  launching no subprocess/stages. These use a mocked IPython display layer, not a Jupyter kernel.
- The source-only upstream contract audit passed against fetched official dataset_readers.py and arguments.
  These were fetched file snapshots, not a local full 4DGS checkout.
- Python source compilation, notebook cell compilation, generator --check, and git diff --check passed.

Not run: PyTorch-dependent tests (including the new rotated pose/TimeMap/preparation integration tests),
real original-EDS loading/preparation, CUDA builds/imports, trained checkpoint loading, GPU render parity,
training, simulation, baseline integrations, or paper metric reproduction. This Windows audit environment
has NumPy but lacks PyTorch, SciPy, OpenCV, h5py, PyYAML and pytest. No dependencies were installed.
Run on the user's existing environment:

```bash
python -m unittest discover -s tests -p 'test_*audit.py' -v
python -m unittest discover -s tests -p test_notebook_contract.py -v
python -m pytest -q tests/
```

## Limits of paper comparison

The paper describes official poses where available, shared Gaussian scene fitting,
and a regular 10x schedule for fixed-budget comparison. It does not establish that
this repository's chosen scene fitting/metric parameters or this selected clip match its full protocol.
Excess rather than raw IG-NLL, Chamfer normalization/voxel/window parameters, Gaussian initialization,
GT frame/image orientation, baseline implementations/parameters and sequence coverage remain material.
ESIM can fall back to repository code; inspect the sidecar's `official_code` flag.
A full reproduction claim also needs matching seeds, windows, sequence list and per-method coverage.
The existing table aggregator averages whatever results it finds; this patch relabels that honestly,
but does not make different protocols comparable.

## Primary sources inspected

- [TIDES v1, sections 4.1–4.3](https://arxiv.org/html/2606.02058v1)
- [Official EDS calibration map](https://github.com/uzh-rpg/bundles-eds/blob/master/config/data/dual_setup/README.md)
- [Official original-recording YAML](https://github.com/uzh-rpg/bundles-eds/blob/master/config/data/dual_setup/03_calib/camchain-mediajaviJAVISdatasetshwdscalibratione2kalibr.yaml)
  blob `edc2962c75ea54fc388b295365d52f03670608d3`.
- [4DGS COLMAP reader](https://github.com/hustvl/4DGaussians/blob/master/scene/dataset_readers.py)
  blob `8a6cc420ac5dae069354ed6b7f099b568235edf9`.
- [4DGS argument definitions](https://github.com/hustvl/4DGaussians/blob/master/arguments/__init__.py)
  blob `7f2e8d6c2b26908c86e93831bb7a971fdcc7afe0`.
- [4DGS training entry point](https://github.com/hustvl/4DGaussians/blob/master/train.py)
  blob `973f8227fac256f9f153b432b392222ecbdfbb5e`.

These SHA values identify file blobs, not pinned upstream repository commits.
The notebook records the user's actual upstream commit at run time.

## Starting with no 4DGS checkpoint (user's current state)

The already prepared 113-frame scene can be reused. The `train` stage fits 4DGS
from scratch: 3000 coarse iterations and 14000 fine iterations. No timing estimate
is asserted without the user's GPU/runtime. Logs stream into the notebook. If training
succeeds, selected simulation/evaluation stages continue in cell order.

After switching to `fix/eds-reproduction-audit`, activate the existing `tides` environment,
set TIDES_FOURDGS to the real installed upstream checkout, and run:

```bash
export TIDES_RUN_STAGES=audit,train,tides,evaluate,record
python -m jupyterlab notebooks/01_eds_paper_reproduction.ipynb
```

Select the `tides` kernel, then Run All. This sequence does not re-prepare the scene and
initially evaluates only the two TIDES methods; unrun baselines and ablations are marked
`not run`. Tables 1, 2 and 4 display ours, paper EDS mean, signed difference and percent
difference. `record` saves CSV and Markdown in `$TIDES_WORK/comparison/` alongside a
reproducibility manifest. This compares the clip with paper dataset means, not equivalent
full-dataset protocols. Values are references transcribed in `tides/paper_numbers.py`.

If all baseline dependencies are installed, include `baselines` between `tides` and
`evaluate`. Ablations are optional and rerun simulation, not 4DGS training.
If dependencies are missing, use `scripts/setup_third_party.sh` after checking the existing
CUDA/PyTorch environment; do not reinstall a working environment blindly.

A terminal-only numerical report is also available:

```bash
python scripts/compare_eds_paper.py --metrics "$TIDES_WORK/metrics.json" --out "$TIDES_WORK/comparison"
```
