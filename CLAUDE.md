# Retrocycles AI

Science-competition (NYSSEF) project: emergent coordination in 7v7 Retrocycles Fortress via PPO self-play.
Successor to the Rocket League project at `C:\Users\allej\Desktop\Rocket League AI` (same experiment
conventions: 3-layer `experiments/params.sh`, `run_label` in metric CSVs, run_all.sh on RunPod).

- Simulator is custom and fully vectorized in torch (`tron/env.py`). Any rule change needs a test in
  `tests/test_env.py` that builds the state by hand and checks the outcome by running it.
  `test_mirror_symmetry_under_mirrored_play` must keep passing. It catches rotation/masking bugs.
- `python -m pytest -q` before every training run.
- No local GPU (CPU torch). Real training happens on RunPod; local runs are smoke tests only.
- Metrics CSV columns are the research data; don't rename them without updating analysis.
