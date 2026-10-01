"""Checks on the trainer's bookkeeping: masks, done flags, credit after death, checkpoints."""
import torch

from tron.config import FortressConfig
from tron.ppo import PPOTrainer, TrainConfig, load_policy
from tron.rewards import RewardConfig, compute_rewards


def tiny(credit=True, pool=0.0):
    g = FortressConfig(team_size=2)
    t = TrainConfig(num_envs=8, rollout_len=40, minibatch=1024, epochs=1, eval_every=0,
                    credit_after_death=credit, pool_frac=pool, pool_every=1)
    return PPOTrainer(g, t, RewardConfig(), "cpu")


def test_credit_after_death_masks():
    tr = tiny(credit=True)
    buf, _, _ = tr.collect()
    p, v, d = buf["pmask"], buf["vmask"], buf["done"]
    assert v.all()                                    # every step trains the value
    assert (p <= v).all()
    assert (~p).any()                                 # some cycles died during 40 random-ish steps
    # done is per round: identical for all agents of an env
    assert (d == d[:, :, :1]).all()


def test_selfish_credit_masks():
    tr = tiny(credit=False)
    buf, _, _ = tr.collect()
    p, v, d, r = buf["pmask"], buf["vmask"], buf["done"], buf["rew"]
    assert torch.equal(p, v)
    # an agent that was already dead before a step gets no reward and is terminal
    dead_steps = ~p
    assert (r[dead_steps] == 0).all()
    assert (d[dead_steps] == 1).all()


def test_dead_cycles_still_get_team_reward_with_credit():
    """Construct a terminal step by hand: team 0 wins while one of its cycles is already dead."""
    T = 2
    info = {
        "done": torch.tensor([True]), "winner": torch.tensor([0]),
        "progress": torch.zeros(1, 2), "progress_before": torch.zeros(1, 2),
        "died": torch.zeros(1, 4, dtype=torch.bool),
    }
    team = torch.tensor([0, 0, 1, 1])
    r, _ = compute_rewards(info, team, T, RewardConfig(), scale=0.0)
    assert r.tolist() == [[1.0, 1.0, -1.0, -1.0]]


def test_shaping_signs():
    T = 1
    info = {
        "done": torch.tensor([False]), "winner": torch.tensor([-1]),
        "progress": torch.tensor([[0.0, 0.1]]), "progress_before": torch.zeros(1, 2),   # team 0 conquering team 1's zone
        "died": torch.tensor([[False, True]]),                                          # team 1's cycle died
    }
    rc = RewardConfig(conquest=1.0, kill=0.1, death=0.1)
    r, _ = compute_rewards(info, torch.tensor([0, 1]), T, rc, scale=1.0)
    assert r[0, 0].item() > 0 and r[0, 1].item() < 0
    assert abs(r[0, 0].item() - (0.1 + 0.1)) < 1e-6
    assert abs(r[0, 1].item() - (-0.1 - 0.1 - 0.1)) < 1e-6


def test_pool_agents_excluded_from_training():
    tr = tiny(pool=0.5)
    tr._snapshot()
    tr._pick_opponent()
    buf, _, _ = tr.collect()
    M, T = tr.n_pool_envs, tr.env.T
    assert M == 4
    assert not buf["vmask"][:, :M, T:].any()
    assert not buf["pmask"][:, :M, T:].any()
    assert buf["vmask"][:, M:, :].all()


def test_iteration_and_checkpoint_roundtrip(tmp_path):
    tr = tiny()
    stats = tr.train_iteration()
    assert stats["samples"] == 40 * 8 * 4
    path = tmp_path / "c.pt"
    torch.save(tr.state(), path)
    model, ck = load_policy(str(path))
    codes, vec = tr.env.observe()
    K = tr.env.K
    with torch.no_grad():
        a, _ = tr.model.eval()(codes.view(-1, K, K), vec.view(-1, tr.env.vec_dim))
        b, _ = model(codes.view(-1, K, K), vec.view(-1, tr.env.vec_dim))
    assert torch.allclose(a, b)
    assert ck["game_cfg"]["team_size"] == 2
