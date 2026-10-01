#!/usr/bin/env bash
# One-time setup on a fresh RunPod GPU pod (any PyTorch template). Run from the repo root.
set -euo pipefail
pip install -q -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible: wrong template or driver mismatch'; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))"
python -m pytest -q tests/test_env.py
python benchmark.py
echo "Now: export NUM_ENVS=<best num_envs from the table above>"
