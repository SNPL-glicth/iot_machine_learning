"""Trajectory and TerminalState domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True, slots=True)
class TerminalState:
    """Terminal state of a trajectory with metadata."""
    state_vector: np.ndarray
    step_index: int
    confidence: float


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Candidate trajectory T of 11–15 movements."""
    movements: tuple[Movement, ...]  # Length 11–15
    coherence_score: float           # Φ_Ritmo(T) from Module 2
    invalidation_step: Optional[int] # Index where trajectory breaks
    terminal_state: TerminalState
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.movements)

    @property
    def delta_states(self) -> np.ndarray:
        """Stack of delta_state vectors for expert evaluation."""
        return np.stack([m.delta_state for m in self.movements])

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([m.timestamp for m in self.movements])

    @property
    def velocities(self) -> np.ndarray:
        return np.array([m.velocity for m in self.movements])

    @property
    def directions(self) -> np.ndarray:
        return np.stack([m.direction for m in self.movements])