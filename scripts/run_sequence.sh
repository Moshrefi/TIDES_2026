#!/usr/bin/env bash
# Full pipeline for one prepared sequence: train 4DGS, simulate TIDES (10x + adaptive),
# run the four baselines on the same renders, and score everything.
#
#   bash scripts/run_sequence.sh work/eds/00_peanuts_dark
#
# Run scripts/prepare_scene.py first to create the work dir.
set -euo pipefail
WORK=$1
FOURDGS=${FOURDGS:-third_party/4DGaussians}
CFG4D=${CFG4D:-configs/4dgs/tides_default.py}
SEED=${SEED:-0}

if [ ! -d "$WORK/4dgs/point_cloud" ]; then
  python "$FOURDGS/train.py" -s "$WORK/scene" -m "$WORK/4dgs" --configs "$CFG4D" --port $((6000 + RANDOM % 1000))
fi
python scripts/simulate.py --work "$WORK" --fourdgs "$FOURDGS" --config configs/tides_10x.yaml --name tides_10x --seed "$SEED"
python scripts/simulate.py --work "$WORK" --fourdgs "$FOURDGS" --config configs/tides_adaptive.yaml --name tides_adaptive --seed "$SEED"
python scripts/run_baselines.py --work "$WORK" --fourdgs "$FOURDGS" --third-party third_party
python scripts/evaluate.py --work "$WORK" --fourdgs "$FOURDGS" \
  --methods esim v2e iebcs dvs_voltmeter tides_10x tides_adaptive
