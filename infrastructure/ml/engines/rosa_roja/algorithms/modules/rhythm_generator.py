"""RhythmTrajectoryGenerator: Phase-space attractor trajectory orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from domain.entities.rosa_roja.movement import Movement
from domain.entities.rosa_roja.trajectory import Trajectory
from domain.entities.rosa_roja.theta_belief import StateKey
from domain.ports.rosa_roja.guided_field import GuidedFieldPort
from domain.entities.rosa_roja.state_persistence import (
    STATE_SCHEMA_VERSION,
    movement_to_raw,
    movements_from_raw,
)
from .theta_belief_manager import ThetaBeliefManager
from .random_walk_sampler import RandomWalkSampler, RandomWalkConfig
from .phi_ritmo_scorer import PhiRitmoScorer


@dataclass
class RhythmTrajectoryGenerator:
    """Trajectory & Continuous Attractor Rhythm Density Generator."""

    min_trajectory_len: int = 11
    max_trajectory_len: int = 15
    max_history_len: int = 200
    top_k: int = 5
    rhythm_weight: float = 0.5
    max_entropy: float = 1.0
    oversample_factor: int = 3
    max_random_walk_steps: int = 100
    invalidation_threshold: float = 0.5
    theta_alpha: float = 0.95
    quantization_decimals: int = 2
    guided_field: Optional[GuidedFieldPort] = None

    def __post_init__(self) -> None:
        self._history: list[Movement] = []
        self._transition_graph: dict[StateKey, list[Movement]] = {}
        self._latest_state_key: Optional[StateKey] = None
        self._symbol: Optional[str] = None
        self._exploration_boost: int = 0

        self._theta_manager = ThetaBeliefManager(theta_alpha=self.theta_alpha, quantization_decimals=self.quantization_decimals)
        self._walk_sampler = RandomWalkSampler(
            config=RandomWalkConfig(
                max_random_walk_steps=self.max_random_walk_steps,
                min_trajectory_len=self.min_trajectory_len,
                max_trajectory_len=self.max_trajectory_len,
                quantization_decimals=self.quantization_decimals,
            ),
            transition_graph=self._transition_graph,
            theta_belief=self._theta_manager.theta,
            quantize_state_func=self._quantize_state,
            guided_field=self.guided_field,
        )
        self._scorer = PhiRitmoScorer(rhythm_weight=self.rhythm_weight, invalidation_threshold=self.invalidation_threshold)

    def set_guided_field(self, guided_field: Optional[GuidedFieldPort]) -> None:
        """Attach or update the GuidedFieldPort provider."""
        self.guided_field = guided_field
        self._walk_sampler.set_guided_field(guided_field)

    # Backward compatibility properties
    @property
    def _theta(self) -> Any:
        return self._theta_manager.theta

    @property
    def _compute_entropy(self) -> Any:
        return self._theta_manager.compute_entropy

    @_compute_entropy.setter
    def _compute_entropy(self, func: Any) -> None:
        self._theta_manager.compute_entropy = func

    @property
    def _random_walk(self) -> Any:
        return self._walk_sampler._random_walk

    @property
    def _phi_ritmo(self) -> Any:
        return self._scorer.score_trajectory

    @property
    def _find_invalidation_step(self) -> Any:
        return self._scorer._find_invalidation_step_vectorized

    @property
    def _compute_transition_weights(self) -> Any:
        return self._walk_sampler.compute_transition_weights

    def _quantize_state(self, state: np.ndarray) -> StateKey:
        arr = np.asarray(state, dtype=np.float64).flatten()
        return tuple(round(float(v), self.quantization_decimals) for v in arr)

    def set_symbol(self, symbol: str) -> None:
        if self._symbol != symbol:
            self.reset()
            self._symbol = symbol

    def reset(self) -> None:
        self._history.clear()
        self._transition_graph.clear()
        self._theta_manager.reset()
        self._latest_state_key = None
        self._exploration_boost = 0

    def boost_exploration(self, events: int) -> None:
        self._exploration_boost = max(0, events)

    def generate_candidate_trajectories(self, latest_movement: Movement, drift_score: float) -> list[Trajectory]:
        self._history.append(latest_movement)
        if len(self._history) > self.max_history_len:
            self._history.pop(0)
        self._update_transition_graph()
        self._latest_state_key = self._quantize_state(latest_movement.delta_state)
        self._theta_manager.update_from_history(self._history)

        if len(self._history) < self.min_trajectory_len:
            return []

        entropy = self._theta_manager.compute_entropy(self._latest_state_key)
        lambda_t = 1.0 if self._exploration_boost > 0 else self._compute_lambda(entropy, drift_score)
        if self._exploration_boost > 0:
            self._exploration_boost -= 1

        candidates = [t for t in self._walk_sampler.generate_candidates(latest_movement, lambda_t, self.top_k * self.oversample_factor) if len(t.movements) >= 2]
        scored = []
        memo: dict[int, Trajectory] = {}
        for traj in candidates:
            tid = id(traj)
            if tid not in memo:
                memo[tid] = self._scorer.score_trajectory(traj, lambda_t, entropy)
            st = memo[tid]
            scored.append((st.coherence_score, st))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[: self.top_k]]

    def _compute_lambda(self, entropy: float, drift_score: float) -> float:
        norm_entropy = min(entropy / self.max_entropy, 1.0) if self.max_entropy > 0 else 0.0
        return min(norm_entropy, max(0.0, 1.0 - drift_score))

    def _update_transition_graph(self) -> None:
        self._transition_graph.clear()
        if len(self._history) < 2:
            self._walk_sampler.update_transition_graph(self._transition_graph)
            return
        for i in range(len(self._history) - 1):
            key = self._quantize_state(self._history[i].delta_state)
            if key not in self._transition_graph:
                self._transition_graph[key] = []
            self._transition_graph[key].append(self._history[i + 1])
        self._walk_sampler.update_transition_graph(self._transition_graph)

    def set_transition_graph(self, graph: dict[StateKey, list[Movement]]) -> None:
        self._transition_graph = graph
        self._walk_sampler.update_transition_graph(graph)

    def export_state(self) -> Dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "history": [movement_to_raw(m) for m in self._history],
            "theta_belief": self._theta_manager.theta.export_state(),
            "exploration_boost": self._exploration_boost,
        }

    def import_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict) or payload.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported payload: {payload.get('schema_version') if isinstance(payload, dict) else type(payload)!r}")
        raw_history = payload.get("history", [])
        self._history = movements_from_raw(raw_history)
        self._update_transition_graph()
        self._theta_manager.theta.import_state(payload.get("theta_belief", {}))
        self._exploration_boost = int(payload.get("exploration_boost", 0))
        self._latest_state_key = self._quantize_state(self._history[-1].delta_state) if self._history else None