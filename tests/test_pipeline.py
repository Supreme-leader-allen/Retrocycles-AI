"""End-to-end checks of the scripts (tiny sizes, CPU): training outputs, resume, PBT."""
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pbt  # noqa: E402
from tron.config import FortressConfig  # noqa: E402
from tron.ppo import PPOTrainer, TrainConfig  # noqa: E402
from tron.rewards import RewardConfig  # noqa: E402
from tron.runner import TrainingRun  # noqa: E402

TINY = ["num_envs=8", "rollout_len=16", "minibatch=512", "epochs=1", "eval_envs=4", "eval_rounds=1",
        "eval_every=2", "replay_every=2", "save_every=2", "check_every=1", "pool_every=1"]


def _tiny_trainer(out, label="t"):
    g = FortressConfig(team_size=1)
    t = TrainConfig(num_envs=8, rollout_len=16, minibatch=512, epochs=1, eval_envs=4, eval_rounds=1,
                    eval_every=2, replay_every=2, save_every=2, check_every=1, pool_every=1)
    tr = PPOTrainer(g, t, RewardConfig(), "cpu", metrics_csv=os.path.join(out, "metrics", f"{label}.csv"),
                    run_label=label)
    return tr, TrainingRun(tr, label, out, verbose=False)


def test_training_run_writes_everything(tmp_path):
    out = str(tmp_path)
    tr, run = _tiny_trainer(out)
    run.run_until(16 * 8 * 2 * 3)          # 3 iterations
    run.close()
    assert tr.iteration == 3
    assert os.path.exists(os.path.join(out, "checkpoints", "t", "latest.pt"))
    log = list(csv.DictReader(open(os.path.join(out, "logs", "t_train.csv"))))
    assert len(log) == 3
    for col in ("act_straight", "rew_win", "rew_conquest", "grad_norm", "side0_win", "end_timeout",
                "deaths_rim", "role_specialization", "eval_heuristic_win", "eval_split_win", "value_mean"):
        assert col in log[0], col
    assert log[0]["eval_heuristic_win"] not in ("", "nan")      # eval ran at iteration 1
    assert os.listdir(os.path.join(out, "heatmaps", "t"))
    assert os.listdir(os.path.join(out, "replays", "t"))
    ev = list(csv.DictReader(open(os.path.join(out, "metrics", "t_eval.csv"))))
    assert {r["run_label"] for r in ev} == {"t_vs_heuristic", "t_vs_split"}


def test_debug_state_saved_on_invariant_violation(tmp_path):
    out = str(tmp_path)
    tr, run = _tiny_trainer(out)
    real = tr.env.check_invariants
    tr.env.check_invariants = lambda: ["forced failure"]
    try:
        run.run_until(10 ** 9)
        assert False, "should have raised"
    except RuntimeError as e:
        assert "forced failure" in str(e)
    run.close()
    tr.env.check_invariants = real
    files = os.listdir(os.path.join(out, "checkpoints", "t"))
    dbg = [f for f in files if f.startswith("debug_invariant")]
    assert dbg
    state = torch.load(os.path.join(out, "checkpoints", "t", dbg[0]), weights_only=False)
    assert "env_state" in state and "owner" in state["env_state"]


def test_pbt_runs_exploits_and_resumes(tmp_path):
    out = str(tmp_path)
    argv = ["--label", "p", "--out", out, "--team-size", "1", "--members", "2", "--generations", "2",
            "--samples", str(2 * 16 * 8 * 2), "--eval-rounds", "4", "--eval-envs", "4", "--device", "cpu",
            "--train"] + TINY
    pbt.main(argv)
    gens = [json.loads(l) for l in open(os.path.join(out, "logs", "p_generations.jsonl"))]
    assert [g["generation"] for g in gens] == [0, 1]
    assert len(gens[0]["replaced"]) == 1                       # bottom member replaced after gen 0
    assert gens[1]["replaced"] == []                           # nothing replaced after the last gen
    rep = gens[0]["replaced"][0]
    assert rep["member"] != rep["copied_from"]
    for k, v in rep["new_hparams"].items():
        parent = gens[0]["members"][rep["copied_from"]]["hparams"][k]
        assert 0.8 * parent - 1e-12 <= v <= 1.2 * parent + 1e-12 or v in pbt.HPARAM_BOUNDS[k]
    assert os.path.exists(os.path.join(out, "checkpoints", "p", "latest.pt"))
    for k in (0, 1):
        ck = torch.load(os.path.join(out, "checkpoints", "p", "members", f"m{k}", "latest.pt"), weights_only=False)
        assert ck["samples"] >= 2 * 16 * 8 * 2
    # re-running is a no-op resume (state says generation 2 = finished)
    pbt.main(argv)
    assert len(open(os.path.join(out, "logs", "p_generations.jsonl")).readlines()) == 2
