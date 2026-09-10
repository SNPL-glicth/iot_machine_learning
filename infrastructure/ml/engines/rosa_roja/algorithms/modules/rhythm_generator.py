"""RhythmTrajectoryGenerator: Main orchestrator for trajectory generation.

Composes ThetaBeliefManager, RandomWalkSampler, and PhiRitmoScorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from ..domain.movement import Movement
from ..domain.trajectory import Trajectory
from ..domain.theta_belief import StateKey
from ..domain.state_persistence import (
    STATE_SCHEMA_VERSION,
    movement_to_raw,
    movements_from_raw,
)
from .theta_belief_manager import ThetaBeliefManager
from .random_walk_sampler import RandomWalkSampler, RandomWalkConfig
from .phi_ritmo_scorer import PhiRitmoScorer


@dataclass
class RhythmTrajectoryGenerator:
    """
    Trajectory & Rhythm Density Generator.
    
    Generates Top-K candidate trajectories T (11 ≤ |T| ≤ 15).
    Normalizes information gain by length |T| to eliminate long-trajectory bias.
    Dynamically adjusts exploration factor λ_t = min(H(Θ|D_t)/H_max, 1 - DriftScore_t).
    
    Equation: Φ_Ritmo(T) = 1/|T| [λ_t · ΔH(T|D_t) + ρ(T)]
    """
    
    min_trajectory_len: int = 11
    max_trajectory_len: int = 15
    top_k: int = 5
    rhythm_weight: float = 0.5
    max_entropy: float = 1.0
    oversample_factor: int = 3
    max_random_walk_steps: int = 100
    invalidation_threshold: float = 0.5
    theta_alpha: float = 0.95
    quantization_decimals: int = 2
    
    def __post_init__(self):
        self._history: list[Movement] = []
        self._transition_graph: dict[StateKey, list[Movement]] = {}
        self._latest_state_key: Optional[StateKey] = None
        self._symbol: Optional[str] = None
        self._exploration_boost: int = 0
        
        # Composed components
        self._theta_manager = ThetaBeliefManager(
            theta_alpha=self.theta_alpha,
            quantization_decimals=self.quantization_decimals,
        )
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
        )
        self._scorer = PhiRitmoScorer(
            rhythm_weight=self.rhythm_weight,
            invalidation_threshold=self.invalidation_threshold,
        )
    
    # Backward compatibility properties for tests
    @property
    def _theta(self):
        """Backward compatibility: access to ThetaBelief."""
        return self._theta_manager.theta
    
    @property
    def _compute_entropy(self):
        """Backward compatibility: delegate to ThetaBeliefManager."""
        return self._theta_manager.compute_entropy
    
    @_compute_entropy.setter
    def _compute_entropy(self, func):
        """Backward compatibility: allow mocking entropy computation."""
        self._theta_manager.compute_entropy = func
    
    @property
    def _random_walk(self):
        """Backward compatibility: delegate to RandomWalkSampler."""
        return self._walk_sampler._random_walk
    
    @property
    def _phi_ritmo(self):
        """Backward compatibility: delegate to PhiRitmoScorer."""
        return self._scorer.score_trajectory
    
    @property
    def _find_invalidation_step(self):
        """Backward compatibility: delegate to PhiRitmoScorer."""
        return self._scorer._find_invalidation_step_vectorized
    
    @property
    def _compute_transition_weights(self):
        """Backward compatibility: delegate to RandomWalkSampler with old signature."""
        return self._walk_sampler.compute_transition_weights
    
    def _quantize_state(self, state: np.ndarray) -> StateKey:
        return tuple(round(v, self.quantization_decimals) for v in state.tolist())
    
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
    
    def generate_candidate_trajectories(
        self,
        latest_movement: Movement,
        drift_score: float,
    ) -> list[Trajectory]:
        """Generate Top-K candidate trajectories from current state."""
        # Update internal history and transition graph
        self._history.append(latest_movement)
        if len(self._history) > 200:
            self._history.pop(0)
        self._update_transition_graph()
        self._latest_state_key = self._quantize_state(latest_movement.delta_state)
        self._theta_manager.update_from_history(self._history)
        
        if len(self._history) < self.min_trajectory_len:
            return []
        
        # Compute exploration factor λ_t
        entropy = self._theta_manager.compute_entropy(self._latest_state_key)
        lambda_t = self._compute_lambda(entropy, drift_score)
        
        if self._exploration_boost > 0:
            lambda_t = 1.0
            self._exploration_boost -= 1
        
        # Generate candidate trajectories. Candidates without at least one
        # predicted step beyond the observed movement (len < 2) carry no
        # predictive horizon and would arm the reactive tracker against a
        # step that cannot be validated.
        candidates = [
            t for t in self._walk_sampler.generate_candidates(
                start_movement=latest_movement,
                lambda_t=lambda_t,
                num_candidates=self.top_k * self.oversample_factor,
            )
            if len(t.movements) >= 2
        ]
        
        # Score and select top-K (memoized by candidate identity: replicated
        # deterministic candidates are scored once)
        scored = []
        score_memo: dict[int, Trajectory] = {}
        for traj in candidates:
            key = id(traj)
            scored_traj = score_memo.get(key)
            if scored_traj is None:
                scored_traj = self._scorer.score_trajectory(traj, lambda_t, entropy)
                score_memo[key] = scored_traj
            scored.append((scored_traj.coherence_score, scored_traj))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:self.top_k]]
    
    def _compute_lambda(self, entropy: float, drift_score: float) -> float:
        normalized_entropy = min(entropy / self.max_entropy, 1.0) if self.max_entropy > 0 else 0.0
        drift_penalty = max(0.0, 1.0 - drift_score)
        return min(normalized_entropy, drift_penalty)
    
    def _update_transition_graph(self) -> None:
        """Rebuild the entire transition graph from history.
        
        This maintains backward compatibility with the original implementation
        which rebuilt the entire graph. The incremental optimization was
        causing issues with tests that manually manipulate history.
        
        Cyclic patterns close themselves naturally: once a state's wrap-around
        transition has been observed at least once, it appears as a regular
        edge. No synthetic edges are fabricated for unseen transitions, since
        guessing a "return to first state" edge introduces phase lag and
        produces phase-shifted trajectories on partially observed cycles.
        """
        self._transition_graph.clear()
        if len(self._history) < 2:
            self._walk_sampler.update_transition_graph(self._transition_graph)
            return
        
        for i in range(len(self._history) - 1):
            curr = self._history[i]
            next_m = self._history[i + 1]
            
            state_key = self._quantize_state(curr.delta_state)
            
            if state_key not in self._transition_graph:
                self._transition_graph[state_key] = []
            self._transition_graph[state_key].append(next_m)
        
        # Update walk sampler's candidate cache with new graph
        self._walk_sampler.update_transition_graph(self._transition_graph)
    
    def set_transition_graph(self, graph: dict[StateKey, list[Movement]]) -> None:
        """Set the transition graph externally (for testing) and update sampler cache."""
        self._transition_graph = graph
        self._walk_sampler.update_transition_graph(graph)

    # ------------------------------------------------------------------
    # Persistence (StatePersistable contract)
    # ------------------------------------------------------------------

    def export_state(self) -> Dict[str, Any]:
        """Serialize learning state to a JSON-safe dict.

        Only the source of truth is persisted: movement history and the
        ThetaBelief posterior. The transition graph is derived state and is
        rebuilt deterministically from history on restore.
        """
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "history": [movement_to_raw(m) for m in self._history],
            "exploration_boost": self._exploration_boost,
            "symbol": self._symbol,
            "theta": self._theta_manager.theta.export_state(),
        }

    def import_state(self, payload: Dict[str, Any]) -> None:
        """Restore learning state from an export_state payload.

        The posterior is imported directly (not replayed): replaying history
        through update() would apply exponential decay twice. The active
        trajectory tracker is intentionally NOT restored — a flushed
        in-flight trajectory yields a safe HOLD on the first post-restore
        event instead of validating against stale predictions.
        """
        if not isinstance(payload, dict) or "schema_version" not in payload:
            raise ValueError("RhythmTrajectoryGenerator payload missing schema_version")
        if payload["schema_version"] != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported RhythmTrajectoryGenerator schema: {payload['schema_version']}"
            )

        history_raw = payload.get("history", [])
        if not isinstance(history_raw, list):
            raise ValueError("RhythmTrajectoryGenerator payload 'history' must be a list")
        for raw in history_raw:
            if not isinstance(raw, dict) or "delta_state" not in raw:
                raise ValueError("Malformed movement entry in history")

        self.reset()
        self._history.extend(movements_from_raw(history_raw))
        self._theta_manager.theta.import_state(payload["theta"])
        self._exploration_boost = int(payload.get("exploration_boost", 0))
        self._symbol = payload.get("symbol")

        # Derive the graph and its sampler cache from restored history.
        self._update_transition_graph()
        if self._history:
            self._latest_state_key = self._quantize_state(
                self._history[-1].delta_state
            )