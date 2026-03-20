"""DQN agent — standard Deep Q-Network with legal action masking."""

from contextlib import nullcontext
import copy
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from src.agents.base_agent import BaseAgent
from src.models.model_factory import build_model


class DQNAgent(BaseAgent):
    """Standard DQN with epsilon-greedy exploration and legal masking."""

    def __init__(self, input_dim: int, num_actions: int, config: Dict[str, Any], device: str = "cpu"):
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.config = config

        agent_cfg = config.get("agent", config)
        self.gamma = agent_cfg.get("discount", {}).get("gamma", 0.99)
        self.grad_clip = agent_cfg.get("training", {}).get("gradient_clip_norm", 10.0)

        # Networks
        self.online_net = build_model(input_dim, num_actions, config).to(self.device)
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        # Optimizer
        lr = agent_cfg.get("optimizer", {}).get("learning_rate", 1e-4)
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Mixed precision (optional)
        train_cfg = agent_cfg.get("training", {})
        amp_enabled = bool(train_cfg.get("mixed_precision", False))
        self._amp_enabled = bool(amp_enabled and self.device.type == "cuda")
        amp_dtype_name = str(train_cfg.get("amp_dtype", "fp16")).lower()
        self._amp_dtype = torch.float16 if amp_dtype_name in {"fp16", "float16", "half"} else torch.bfloat16
        self._scaler = GradScaler(enabled=self._amp_enabled)

        # Exploration
        exp_cfg = agent_cfg.get("exploration", {})
        self.epsilon = exp_cfg.get("epsilon_start", 1.0)
        self.epsilon_end = exp_cfg.get("epsilon_end", 0.01)
        self.epsilon_decay_steps = exp_cfg.get("epsilon_decay_steps", 100000)
        self._step_count = 0

        # Target update
        self.target_update_interval = agent_cfg.get("target", {}).get("update_interval", 1000)
        self._learn_count = 0

    def act(self, observation: np.ndarray, mask: np.ndarray, training: bool = True) -> int:
        legal_actions = np.where(mask > 0)[0]
        if len(legal_actions) == 0:
            return 0  # HOLD fallback

        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(legal_actions))

        with torch.inference_mode():
            obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(obs_t).squeeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            q_values = q_values.masked_fill(~mask_t, -torch.inf)
            return int(torch.argmax(q_values).item())

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=self.device)

        self.optimizer.zero_grad(set_to_none=True)
        amp_ctx = autocast(dtype=self._amp_dtype, enabled=self._amp_enabled) if self.device.type == "cuda" else nullcontext()

        with amp_ctx:
            q_values = self.online_net(obs)
            q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q = self.target_net(next_obs)
                fill = torch.finfo(next_q.dtype).min
                next_q = next_q.masked_fill(~next_masks, fill)
                max_next_q = next_q.max(dim=1).values
                targets = rewards + self.gamma * (1 - dones) * max_next_q

            loss = self.loss_fn(q_selected, targets)

        if self._amp_enabled:
            self._scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
            self.optimizer.step()

        self._learn_count += 1
        if self._learn_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Update epsilon
        self._step_count += 1
        self.epsilon = max(
            self.epsilon_end,
            1.0 - (1.0 - self.epsilon_end) * self._step_count / max(self.epsilon_decay_steps, 1),
        )

        return {"loss": float(loss.item()), "epsilon": self.epsilon, "q_mean": float(q_selected.mean().item())}

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self._step_count,
            "learn_count": self._learn_count,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self._step_count = checkpoint.get("step_count", 0)
        self._learn_count = checkpoint.get("learn_count", 0)

    def train_mode(self) -> None:
        self.online_net.train()

    def eval_mode(self) -> None:
        self.online_net.eval()
