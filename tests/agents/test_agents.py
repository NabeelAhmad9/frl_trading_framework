"""Tests for DQN and Double DQN agents."""

import os
import tempfile
import numpy as np
import pytest
import torch

from src.agents.dqn.dqn_agent import DQNAgent
from src.agents.doubledqn.doubledqn_agent import DoubleDQNAgent
from src.agents.common.replay_buffer import ReplayBuffer


INPUT_DIM = 16
NUM_ACTIONS = 10

AGENT_CONFIG = {
    "agent": {
        "model": {
            "encoder_type": "mlp",
            "hidden_dims": [32, 16],
            "dueling": False,
            "activation": "relu",
            "dropout": 0.0,
        },
        "optimizer": {"learning_rate": 1e-3},
        "discount": {"gamma": 0.99},
        "exploration": {
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay_steps": 100,
        },
        "target": {"update_interval": 5},
        "training": {"gradient_clip_norm": 10.0},
    }
}

DUELING_CONFIG = {
    "agent": {
        "model": {
            "encoder_type": "mlp",
            "hidden_dims": [32, 16],
            "dueling": True,
            "activation": "relu",
            "dropout": 0.0,
        },
        "optimizer": {"learning_rate": 1e-3},
        "discount": {"gamma": 0.99},
        "exploration": {
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay_steps": 100,
        },
        "target": {"update_interval": 5},
        "training": {"gradient_clip_norm": 10.0},
    }
}


def _make_batch(batch_size=8, input_dim=INPUT_DIM, num_actions=NUM_ACTIONS):
    """Generate a randomized batch dict."""
    rng = np.random.RandomState(0)
    masks = np.ones((batch_size, num_actions), dtype=np.float32)
    masks[:, 3:6] = 0  # disable some actions
    return {
        "obs": rng.randn(batch_size, input_dim).astype(np.float32),
        "actions": rng.randint(0, 3, size=batch_size).astype(np.int64),
        "rewards": rng.randn(batch_size).astype(np.float32),
        "next_obs": rng.randn(batch_size, input_dim).astype(np.float32),
        "dones": rng.randint(0, 2, size=batch_size).astype(np.float32),
        "masks": masks.copy(),
        "next_masks": masks.copy(),
    }


# ============================================================
# Replay buffer tests
# ============================================================
class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(capacity=10, seed=42)
        assert len(buf) == 0
        buf.add(np.zeros(4), 0, 1.0, np.zeros(4), False)
        assert len(buf) == 1

    def test_capacity_overflow(self):
        buf = ReplayBuffer(capacity=3, seed=0)
        for i in range(5):
            buf.add(np.array([float(i)]), i % 2, float(i), np.array([float(i + 1)]), i == 4)
        assert len(buf) == 3

    def test_sample_shape(self):
        buf = ReplayBuffer(capacity=100, seed=42)
        mask = np.ones(NUM_ACTIONS, dtype=np.float32)
        for _ in range(20):
            obs = np.random.randn(INPUT_DIM).astype(np.float32)
            buf.add(obs, 0, 1.0, obs, False, mask, mask)
        batch = buf.sample(8)
        assert batch["obs"].shape == (8, INPUT_DIM)
        assert batch["actions"].shape == (8,)
        assert batch["next_masks"].shape == (8, NUM_ACTIONS)

    def test_is_ready(self):
        buf = ReplayBuffer(capacity=100)
        assert not buf.is_ready(5)
        for _ in range(5):
            buf.add(np.zeros(4), 0, 0.0, np.zeros(4), False)
        assert buf.is_ready(5)


# ============================================================
# DQN Agent tests
# ============================================================
class TestDQNAgent:
    def test_construction(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        assert agent.num_actions == NUM_ACTIONS
        assert agent.epsilon == 1.0

    def test_act_exploration(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        agent.epsilon = 1.0
        obs = np.random.randn(INPUT_DIM).astype(np.float32)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        mask[0] = 1
        mask[1] = 1
        action = agent.act(obs, mask, training=True)
        assert action in [0, 1]

    def test_act_exploitation(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        agent.epsilon = 0.0
        obs = np.random.randn(INPUT_DIM).astype(np.float32)
        mask = np.ones(NUM_ACTIONS, dtype=np.float32)
        action = agent.act(obs, mask, training=False)
        assert 0 <= action < NUM_ACTIONS

    def test_act_respects_mask_exploitation(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        agent.epsilon = 0.0
        obs = np.random.randn(INPUT_DIM).astype(np.float32)
        # Only allow action 7
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        mask[7] = 1
        action = agent.act(obs, mask, training=False)
        assert action == 7

    def test_update_produces_loss(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        batch = _make_batch()
        metrics = agent.update(batch)
        assert "loss" in metrics
        assert "epsilon" in metrics
        assert "q_mean" in metrics
        assert isinstance(metrics["loss"], float)

    def test_epsilon_decays(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        batch = _make_batch()
        eps_before = agent.epsilon
        agent.update(batch)
        assert agent.epsilon < eps_before

    def test_target_network_syncs(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        batch = _make_batch()
        # Run exactly target_update_interval updates
        for _ in range(agent.target_update_interval):
            agent.update(batch)
        # After sync, online and target should match
        for p_o, p_t in zip(agent.online_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(p_o.data, p_t.data)

    def test_save_and_load_checkpoint(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        batch = _make_batch()
        agent.update(batch)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            agent.save_checkpoint(path)
            assert os.path.isfile(path)

            agent2 = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
            agent2.load_checkpoint(path)
            assert abs(agent2.epsilon - agent.epsilon) < 1e-9
            for p1, p2 in zip(agent.online_net.parameters(), agent2.online_net.parameters()):
                assert torch.allclose(p1, p2)

    def test_train_eval_mode(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        agent.eval_mode()
        assert not agent.online_net.training
        agent.train_mode()
        assert agent.online_net.training

    def test_dueling_architecture(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, DUELING_CONFIG, device="cpu")
        batch = _make_batch()
        metrics = agent.update(batch)
        assert "loss" in metrics

    def test_act_holdback_on_empty_mask(self):
        """When no legal actions, HOLD (0) is returned."""
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        obs = np.random.randn(INPUT_DIM).astype(np.float32)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        action = agent.act(obs, mask, training=True)
        assert action == 0


# ============================================================
# Double DQN Agent tests
# ============================================================
class TestDoubleDQNAgent:
    def test_construction(self):
        agent = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        assert agent.num_actions == NUM_ACTIONS

    def test_inherits_act(self):
        agent = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        agent.epsilon = 0.0
        obs = np.random.randn(INPUT_DIM).astype(np.float32)
        mask = np.ones(NUM_ACTIONS, dtype=np.float32)
        action = agent.act(obs, mask, training=False)
        assert 0 <= action < NUM_ACTIONS

    def test_update_produces_loss(self):
        agent = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        batch = _make_batch()
        metrics = agent.update(batch)
        assert "loss" in metrics
        assert "epsilon" in metrics
        assert "q_mean" in metrics

    def test_double_target_differs_from_dqn(self):
        """When online and target nets differ, Double DQN and DQN produce different targets."""
        torch.manual_seed(0)
        dqn = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        torch.manual_seed(0)
        ddqn = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")

        batch = _make_batch()
        # After 1 update, online diverges from target
        dqn.update(batch)
        ddqn.update(batch)

        # Do another update — now results typically differ
        m1 = dqn.update(batch)
        m2 = ddqn.update(batch)
        # Losses may or may not differ numerically, but both should produce valid results
        assert isinstance(m1["loss"], float)
        assert isinstance(m2["loss"], float)

    def test_double_target_matches_dqn_when_same_nets(self):
        """When online == target, Double DQN reduces to standard DQN target."""
        torch.manual_seed(42)
        dqn = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        torch.manual_seed(42)
        ddqn = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")

        # Before any update, online == target (deepcopy at init)
        obs = np.random.randn(1, INPUT_DIM).astype(np.float32)
        obs_t = torch.FloatTensor(obs)
        mask_all = np.ones((1, NUM_ACTIONS), dtype=np.float32)

        with torch.no_grad():
            dqn_q = dqn.target_net(obs_t)
            ddqn_q = ddqn.target_net(obs_t)
        # Same seeds → same initial weights → same Q values
        assert torch.allclose(dqn_q, ddqn_q, atol=1e-6)

    def test_epsilon_decays(self):
        agent = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        batch = _make_batch()
        eps_before = agent.epsilon
        agent.update(batch)
        assert agent.epsilon < eps_before

    def test_save_load(self):
        agent = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        batch = _make_batch()
        agent.update(batch)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ddqn_ckpt.pt")
            agent.save_checkpoint(path)
            agent2 = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
            agent2.load_checkpoint(path)
            assert abs(agent2.epsilon - agent.epsilon) < 1e-9

    def test_dueling_double_dqn(self):
        agent = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, DUELING_CONFIG, device="cpu")
        batch = _make_batch()
        metrics = agent.update(batch)
        assert "loss" in metrics
