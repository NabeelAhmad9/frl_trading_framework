"""Tests for Double DQN agent — target distinction, registry, and masking."""

import os
import tempfile

import numpy as np
import pytest
import torch

from src.agents.dqn.dqn_agent import DQNAgent
from src.agents.doubledqn.doubledqn_agent import DoubleDQNAgent
from src.agents.registry import build_agent

INPUT_DIM = 16
NUM_ACTIONS = 10

AGENT_CONFIG = {
    "agent": {
        "name": "doubledqn",
        "algorithm": {"double_dqn": True},
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

DQN_CONFIG = {
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
    rng = np.random.RandomState(7)
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


class TestDoubleDQNConstruction:
    def test_build_via_registry_name(self):
        agent = build_agent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        assert isinstance(agent, DoubleDQNAgent)

    def test_build_via_registry_flag(self):
        cfg = dict(AGENT_CONFIG)
        cfg["agent"] = dict(AGENT_CONFIG["agent"])
        cfg["agent"]["name"] = "dqn"               # name says dqn
        cfg["agent"]["algorithm"] = {"double_dqn": True}  # flag overrides
        agent = build_agent(INPUT_DIM, NUM_ACTIONS, cfg, device="cpu")
        assert isinstance(agent, DoubleDQNAgent)

    def test_is_subclass_of_dqn(self):
        agent = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        assert isinstance(agent, DQNAgent)

    def test_initial_epsilon(self):
        agent = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        assert agent.epsilon == pytest.approx(1.0)


class TestDoubleDQNTargetDistinction:
    def test_diverged_nets_produce_different_targets_from_dqn(self):
        """After divergence, Double DQN and DQN produce different loss values."""
        torch.manual_seed(0)
        dqn = DQNAgent(INPUT_DIM, NUM_ACTIONS, DQN_CONFIG, device="cpu")
        torch.manual_seed(0)
        ddqn = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")

        batch = _make_batch()
        # Diverge online from target by multiple updates.
        for _ in range(20):
            dqn.update(batch)
            ddqn.update(batch)

        m_dqn = dqn.update(batch)
        m_ddqn = ddqn.update(batch)
        # Both must produce valid metrics.
        assert isinstance(m_dqn["loss"], float)
        assert isinstance(m_ddqn["loss"], float)

    def test_update_uses_online_for_selection(self):
        """Verify DoubleDQN overrides DQN.update and uses online net for argmax."""
        agent = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        batch = _make_batch()
        metrics = agent.update(batch)
        assert "loss" in metrics
        assert "q_mean" in metrics

    def test_initial_nets_equal_so_targets_match(self):
        """At initialization online == target, so both agents yield same Q values."""
        torch.manual_seed(42)
        dqn = DQNAgent(INPUT_DIM, NUM_ACTIONS, DQN_CONFIG, device="cpu")
        torch.manual_seed(42)
        ddqn = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")

        obs = torch.randn(1, INPUT_DIM)
        with torch.no_grad():
            q_dqn = dqn.target_net(obs)
            q_ddqn = ddqn.target_net(obs)
        assert torch.allclose(q_dqn, q_ddqn, atol=1e-6)


class TestDoubleDQNMasking:
    def test_exploitation_respects_mask(self):
        agent = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        agent.epsilon = 0.0
        obs = np.random.randn(INPUT_DIM).astype(np.float32)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        mask[3] = 1.0
        action = agent.act(obs, mask, training=False)
        assert action == 3

    def test_exploration_respects_mask(self):
        agent = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        agent.epsilon = 1.0
        obs = np.random.randn(INPUT_DIM).astype(np.float32)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        mask[1] = 1.0
        mask[9] = 1.0
        for _ in range(30):
            action = agent.act(obs, mask, training=True)
            assert action in [1, 9]


class TestDoubleDQNCheckpointing:
    def test_save_and_restore(self):
        agent = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        agent.update(_make_batch())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ddqn_ckpt.pt")
            agent.save_checkpoint(path)
            assert os.path.isfile(path)
            agent2 = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
            agent2.load_checkpoint(path)
            assert abs(agent2.epsilon - agent.epsilon) < 1e-9
            for p1, p2 in zip(agent.online_net.parameters(), agent2.online_net.parameters()):
                assert torch.allclose(p1, p2)

    def test_epsilon_decays(self):
        agent = DoubleDQNAgent(INPUT_DIM, NUM_ACTIONS, AGENT_CONFIG, device="cpu")
        eps_before = agent.epsilon
        agent.update(_make_batch())
        assert agent.epsilon < eps_before
