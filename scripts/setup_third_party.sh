#!/usr/bin/env bash
# Clone and install the external code TIDES builds on. Run from the repo root,
# inside the Python environment you will use (CUDA toolkit must match PyTorch).
set -euo pipefail
mkdir -p third_party && cd third_party

# 4D Gaussian Splatting (Wu et al., CVPR 2024): scene fitting
if [ ! -d 4DGaussians ]; then
  git clone --recursive https://github.com/hustvl/4DGaussians
fi
pip install -r 4DGaussians/requirements.txt
pip install -e 4DGaussians/submodules/depth-diff-gaussian-rasterization
pip install -e 4DGaussians/submodules/simple-knn

# Baselines
[ -d rpg_vid2e ] || git clone https://github.com/uzh-rpg/rpg_vid2e
pip install rpg_vid2e/esim_torch/ || echo "esim_torch build failed: ESIM will use the built-in re-implementation"
pip install v2e || (git clone https://github.com/SensorsINI/v2e && pip install -e v2e)
[ -d DVS-Voltmeter ] || git clone https://github.com/Lynn0306/DVS-Voltmeter
pip install easydict
[ -d IEBCS ] || git clone https://github.com/neuromorphicsystems/IEBCS
echo "third-party setup done"
