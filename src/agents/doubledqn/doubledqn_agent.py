"""Double DQN agent — online network selects action, target network evaluates."""

from contextlib import nullcontext

import torch
import numpy as np
from typing import Any, Dict

from src.agents.dqn.dqn_agent import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """Double DQN: uses online net for action selection, target net for evaluation."""

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=self.device)

        self.optimizer.zero_grad(set_to_none=True)
        amp_ctx = torch.cuda.amp.autocast(dtype=self._amp_dtype, enabled=self._amp_enabled) if self.device.type == "cuda" else nullcontext()

        with amp_ctx:
            q_values = self.online_net(obs)
            q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_online = self.online_net(next_obs)
                fill = torch.finfo(next_q_online.dtype).min
                next_q_online = next_q_online.masked_fill(~next_masks, fill)
                best_actions = next_q_online.argmax(dim=1)

                next_q_target = self.target_net(next_obs)
                max_next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + self.gamma * (1 - dones) * max_next_q

            loss = self.loss_fn(q_selected, targets)

        if self._amp_enabled:
            self._scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
            self.optimizer.step()

        self._learn_count += 1
        if self._learn_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self._step_count += 1
        self.epsilon = max(
            self.epsilon_end,
            1.0 - (1.0 - self.epsilon_end) * self._step_count / max(self.epsilon_decay_steps, 1),
        )

        return {"loss": float(loss.item()), "epsilon": self.epsilon, "q_mean": float(q_selected.mean().item())}
