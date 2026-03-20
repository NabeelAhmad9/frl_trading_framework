"""Epsilon scheduler — exploration annealing strategies."""


from typing import Dict


class EpsilonScheduler:
    """Linear epsilon decay with extensibility for other schedules."""

    def __init__(self, start: float = 1.0, end: float = 0.01, decay_steps: int = 100000):
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = max(int(decay_steps), 1)
        self._step = 0
        self.epsilon = self.start

    def epsilon_at(self, step: int) -> float:
        """Return epsilon value at an arbitrary step without mutating state."""
        step = max(int(step), 0)
        frac = min(step / self.decay_steps, 1.0)
        return self.start + (self.end - self.start) * frac

    def step(self) -> float:
        """Advance one step and return current epsilon."""
        self._step += 1
        self.epsilon = self.epsilon_at(self._step)
        return self.epsilon

    def set_step(self, step: int) -> None:
        """Restore step counter (for resume)."""
        self._step = max(int(step), 0)
        self.epsilon = self.epsilon_at(self._step)

    def state_dict(self) -> Dict[str, float]:
        """Return serializable scheduler state."""
        return {
            "start": self.start,
            "end": self.end,
            "decay_steps": self.decay_steps,
            "step": self._step,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: Dict[str, float]) -> None:
        """Load scheduler state from checkpoint payload."""
        if not isinstance(state, dict):
            return

        self.start = float(state.get("start", self.start))
        self.end = float(state.get("end", self.end))
        self.decay_steps = max(int(state.get("decay_steps", self.decay_steps)), 1)
        self.set_step(int(state.get("step", self._step)))
        if "epsilon" in state:
            self.epsilon = float(state["epsilon"])

    @property
    def current_step(self) -> int:
        return self._step
