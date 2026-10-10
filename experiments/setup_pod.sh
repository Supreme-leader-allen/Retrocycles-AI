#!/usr/bin/env bash
# One-time setup on a fresh RunPod GPU pod (any PyTorch template). Run from the repo root.
set -euo pipefail
pip install -q -r requirements.txt
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not visible: wrong template or driver mismatch"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0),
      "| %.0f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
# actually run GPU kernels: a newer GPU on too old a torch build fails here ("no kernel image")
x = torch.randn(64, 8, 21, 21, device="cuda")
conv = torch.nn.Conv2d(8, 32, 3, padding=1).cuda()
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = conv(x).sum()
y.backward()
torch.cuda.synchronize()
print("GPU kernels and bfloat16 OK")
PY
df -h /workspace | tail -1 | awk '{print "free disk on /workspace: " $4}'
python -m pytest -q tests/test_env.py
python benchmark.py
echo "Now: export NUM_ENVS=<best num_envs from the table above>"
