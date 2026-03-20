"""Replay buffer — store transitions and sample mini-batches."""

from typing import Any, Dict, Optional

import numpy as np


class ReplayBuffer:
    """Fixed-size circular replay buffer with legal mask support."""

    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = int(capacity)
        self.rng = np.random.RandomState(seed)
        # Legacy field kept for backwards compatibility with historical runtime_state payloads.
        self._storage = []
        self._pos = 0
        self._size = 0

        # Lazily allocated contiguous arrays (allocated on first insert).
        self._obs: Optional[np.ndarray] = None
        self._actions: Optional[np.ndarray] = None
        self._rewards: Optional[np.ndarray] = None
        self._next_obs: Optional[np.ndarray] = None
        self._dones: Optional[np.ndarray] = None
        self._masks: Optional[np.ndarray] = None
        self._next_masks: Optional[np.ndarray] = None
        self._obs_shape: Optional[tuple] = None
        self._mask_shape: Optional[tuple] = None

    def __len__(self):
        return int(self._size)

    def _allocate(self, obs: np.ndarray, mask: np.ndarray) -> None:
        obs_arr = np.asarray(obs, dtype=np.float32)
        mask_arr = np.asarray(mask, dtype=np.float32)

        self._obs_shape = tuple(obs_arr.shape)
        self._mask_shape = tuple(mask_arr.shape)

        self._obs = np.empty((self.capacity, *self._obs_shape), dtype=np.float32)
        self._actions = np.empty((self.capacity,), dtype=np.int64)
        self._rewards = np.empty((self.capacity,), dtype=np.float32)
        self._next_obs = np.empty((self.capacity, *self._obs_shape), dtype=np.float32)
        self._dones = np.empty((self.capacity,), dtype=np.float32)
        self._masks = np.empty((self.capacity, *self._mask_shape), dtype=np.float32)
        self._next_masks = np.empty((self.capacity, *self._mask_shape), dtype=np.float32)

    @staticmethod
    def _default_mask(next_obs: np.ndarray) -> np.ndarray:
        # Retain historical fallback behavior for transitions with missing mask data.
        _ = next_obs
        return np.ones(1, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        mask: Optional[np.ndarray] = None,
        next_mask: Optional[np.ndarray] = None,
    ) -> None:
        if mask is None:
            mask = self._default_mask(obs)
        if next_mask is None:
            next_mask = self._default_mask(next_obs)

        if self._obs is None:
            self._allocate(obs, mask)

        assert self._obs is not None
        assert self._actions is not None
        assert self._rewards is not None
        assert self._next_obs is not None
        assert self._dones is not None
        assert self._masks is not None
        assert self._next_masks is not None

        self._obs[self._pos] = np.asarray(obs, dtype=np.float32)
        self._actions[self._pos] = int(action)
        self._rewards[self._pos] = float(reward)
        self._next_obs[self._pos] = np.asarray(next_obs, dtype=np.float32)
        self._dones[self._pos] = float(done)
        self._masks[self._pos] = np.asarray(mask, dtype=np.float32)
        self._next_masks[self._pos] = np.asarray(next_mask, dtype=np.float32)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self._size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        assert self._obs is not None
        assert self._actions is not None
        assert self._rewards is not None
        assert self._next_obs is not None
        assert self._dones is not None
        assert self._masks is not None
        assert self._next_masks is not None

        indices = self.rng.randint(0, self._size, size=int(batch_size))

        return {
            "obs": self._obs[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "next_obs": self._next_obs[indices],
            "dones": self._dones[indices],
            "masks": self._masks[indices],
            "next_masks": self._next_masks[indices],
        }

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= int(batch_size)

    def state_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "capacity": int(self.capacity),
            "pos": int(self._pos),
            "size": int(self._size),
            "rng_state": self.rng.get_state(),
        }

        if self._size <= 0 or self._obs is None:
            payload.update({
                "obs": None,
                "actions": None,
                "rewards": None,
                "next_obs": None,
                "dones": None,
                "masks": None,
                "next_masks": None,
                "storage": [],
            })
            return payload

        payload.update({
            "obs": self._obs[: self._size].copy(),
            "actions": self._actions[: self._size].copy(),
            "rewards": self._rewards[: self._size].copy(),
            "next_obs": self._next_obs[: self._size].copy(),
            "dones": self._dones[: self._size].copy(),
            "masks": self._masks[: self._size].copy(),
            "next_masks": self._next_masks[: self._size].copy(),
            # Keep legacy key so old checkpoints/tools can still inspect presence.
            "storage": [],
        })
        return payload

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not isinstance(state, dict):
            return

        self._pos = 0
        self._size = 0
        self._obs = None
        self._actions = None
        self._rewards = None
        self._next_obs = None
        self._dones = None
        self._masks = None
        self._next_masks = None
        self._obs_shape = None
        self._mask_shape = None
        self._storage = []

        arrays_present = state.get("obs") is not None and state.get("next_obs") is not None
        if arrays_present:
            obs = np.asarray(state.get("obs"), dtype=np.float32)
            actions = np.asarray(state.get("actions"), dtype=np.int64)
            rewards = np.asarray(state.get("rewards"), dtype=np.float32)
            next_obs = np.asarray(state.get("next_obs"), dtype=np.float32)
            dones = np.asarray(state.get("dones"), dtype=np.float32)
            masks = np.asarray(state.get("masks"), dtype=np.float32)
            next_masks = np.asarray(state.get("next_masks"), dtype=np.float32)

            size = int(state.get("size", len(obs)))
            size = max(0, min(size, len(obs), self.capacity))
            if size > 0:
                self._allocate(obs[0], masks[0])
                assert self._obs is not None
                assert self._actions is not None
                assert self._rewards is not None
                assert self._next_obs is not None
                assert self._dones is not None
                assert self._masks is not None
                assert self._next_masks is not None

                self._obs[:size] = obs[:size]
                self._actions[:size] = actions[:size]
                self._rewards[:size] = rewards[:size]
                self._next_obs[:size] = next_obs[:size]
                self._dones[:size] = dones[:size]
                self._masks[:size] = masks[:size]
                self._next_masks[:size] = next_masks[:size]
                self._size = size
                self._pos = int(state.get("pos", size)) % max(self.capacity, 1)
        else:
            # Backward compatibility: historical checkpoints stored list-based transitions.
            storage = state.get("storage", [])
            if isinstance(storage, list):
                for transition in storage[-self.capacity :]:
                    if not isinstance(transition, (list, tuple)) or len(transition) < 5:
                        continue
                    obs, action, reward, next_obs, done = transition[:5]
                    mask = transition[5] if len(transition) > 5 else None
                    next_mask = transition[6] if len(transition) > 6 else None
                    self.add(obs, action, reward, next_obs, done, mask, next_mask)
            self._pos = int(state.get("pos", self._size)) % max(self.capacity, 1)

        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.set_state(rng_state)
