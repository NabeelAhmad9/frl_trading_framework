"""Tests for DQN agent — legal masking, target logic, checkpointing."""

import os
import tempfile

import numpy as np
import pytest
import torch

from src.agents.dqn.dqn_agent import DQNAgent
from src.agents.registry import build_agent

INPUT_DIM = 16
NUM_ACTIONS = 10

AGENT_CONFIG = {
    "agent": {
        "name": "dqn",
        "algorithm": {"double_dqn": False},
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


def _make_batch(batch_size: int = 8):
    rng = np.random.RandomState(0)
    masks = np.ones((batch_size, NUM_ACTIONS), dtype=np.float32)
    masks[:, 5:8] = 0
    return {
        "obs": rng.randn(batch_size, INPUT_DIM).astype(np.float32),
        "actions": rng.randint(0, 3, size=batch_size).astype(np.int64),
        "rewards": rng.randn(batch_size).astype(np.float32),
        "next_obs": rng.randn(batch_size, INPUT_DIM).astype(np.float32),
        "dones": rng.randint(0, 2, size=batch_size).astype(np.float32),
        "masks": masks.copy(),
        "next_masks": masks.copy(),
    }


class TestDQNAgentConstruction:
    def test_build_via_registry(self):
        agent = build_agent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        assert isinstance(agent, DQNAgent)
        assert agent.num_actions == NUM_ACTIONS

    def test_initial_epsilon(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        assert agent.epsilon == pytest.approx(1.0)

    def test_action_dimension(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        obs = np.random.randn(INPUT_DIM).astype(np.float32)
        mask = np.ones(NUM_ACTIONS, dtype=np.float32)
        action = agent.act(obs, mask, training=False)
        assert 0 <= action < NUM_ACTIONS


class TestDQNLegalMasking:
    def test_exploration_respects_mask(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        agent.epsilon = 1.0
        obs = np.random.randn(INPUT_DIM).astype(np.float32)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        mask[2] = 1.0
        mask[4] = 1.0
        for _ in range(50):
            action = agent.act(obs, mask, training=True)
            assert action in [2, 4]

    def test_exploitation_respects_mask(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        agent.epsilon = 0.0
        obs = np.random.randn(INPUT_DIM).astype(np.float32)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        mask[7] = 1.0
        action = agent.act(obs, mask, training=False)
        assert action == 7

    def test_empty_mask_fallback(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        obs = np.random.randn(INPUT_DIM).astype(np.float32)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        action = agent.act(obs, mask, training=True)
        assert action == 0


class TestDQNTargetLogic:
    def test_update_produces_metrics(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        metrics = agent.update(_make_batch())
        assert "loss" in metrics
        assert "epsilon" in metrics
        assert "q_mean" in metrics
        assert isinstance(metrics["loss"], float)

    def test_target_syncs_at_interval(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        batch = _make_batch()
        for _ in range(agent.target_update_interval):
            agent.update(batch)
        for po, pt in zip(agent.online_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(po.data, pt.data)

    def test_epsilon_decays_after_update(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        eps_start = agent.epsilon
        agent.update(_make_batch())
        assert agent.epsilon < eps_start


class TestDQNCheckpointing:
    def test_save_and_restore(self):
        agent = DQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        agent.update(_make_batch())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dqn_ckpt.pt")
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
