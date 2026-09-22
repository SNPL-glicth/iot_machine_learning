"""RandomWalkSampler: Continuous phase-space candidate trajectory generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from ..domain.movement import Movement
from ..domain.trajectory import Trajectory, TerminalState
from ..domain.theta_belief import StateKey


@dataclass
class RandomWalkConfig:
    max_random_walk_steps: int = 100
    min_trajectory_len: int = 11
    max_trajectory_len: int = 15
    quantization_decimals: int = 2
    rbf_bandwidth: float = 0.5


class RandomWalkSampler:
    """Generates trajectories along manifold flow without fabricating phase-shifted transitions."""

    def __init__(self, config: RandomWalkConfig, transition_graph: dict[StateKey, list[Movement]], theta_belief: Any, quantize_state_func: Any) -> None:
        self._config = config
        self._transition_graph = transition_graph
        self._theta = theta_belief
        self._quantize_state = quantize_state_func
        self._graph_version = 0
        self._cache_version = -1
        self._candidate_cache: dict[Any, dict[str, Any]] = {}
        self._precompute_candidate_data()

    def update_transition_graph(self, transition_graph: dict[StateKey, list[Movement]]) -> None:
        self._transition_graph = transition_graph
        self._graph_version += 1
        self._precompute_candidate_data()

    def _ensure_cache_valid(self) -> None:
        if self._cache_version != self._graph_version:
            self._cache_version = self._graph_version
            self._precompute_candidate_data()

    def _precompute_candidate_data(self) -> None:
        self._candidate_cache.clear()
        for key, movements in self._transition_graph.items():
            if not movements:
                continue
            self._candidate_cache[key] = {
                "movements": np.array(movements, dtype=object),
                "velocities": np.array([m.velocity for m in movements], dtype=np.float64),
                "directions": np.array([m.direction for m in movements], dtype=np.float64),
                "tempos": np.array([m.rhythm_signature.tempo_ratio for m in movements], dtype=np.float64),
                "delta_states": np.array([m.delta_state for m in movements], dtype=np.float64),
                "quantized_keys": tuple(self._quantize_state(m.delta_state) for m in movements),
                "n": len(movements),
            }

    def _compute_transition_weights_cached(self, from_vel: float, from_dir: np.ndarray, from_tempo: float, cache: dict, posterior: Optional[dict], lambda_t: float) -> np.ndarray:
        n = cache["n"]
        if n == 1:
            return np.array([1.0])
        uniform = np.full(n, 1.0 / n)
        vel_sim = np.maximum(0.0, 1.0 - np.abs(cache["velocities"] - from_vel) / (abs(from_vel) + 1e-6))
        dir_sim = np.maximum(0.0, cache["directions"] @ from_dir)
        tempo_sim = np.where((from_tempo > 1e-6) & (cache["tempos"] > 1e-6), 1.0 - np.minimum(1.0, np.abs(np.log(cache["tempos"] / from_tempo))), 0.5)
        coherence = np.maximum(0.0, 0.4 * vel_sim + 0.4 * dir_sim + 0.2 * tempo_sim)
        c_sum = coherence.sum()
        c_dist = coherence / c_sum if c_sum > 0 else uniform.copy()

        if posterior:
            p_arr = np.array([posterior.get(k, 0.0) for k in cache["quantized_keys"]])
            p_sum = p_arr.sum()
            p_dist = p_arr / p_sum if p_sum > 0 else uniform.copy()
        else:
            p_dist = uniform.copy()

        emp = p_dist * c_dist
        emp_sum = emp.sum()
        emp_dist = emp / emp_sum if emp_sum > 0 else uniform.copy()
        lam = float(np.clip(lambda_t, 0.0, 1.0))
        return np.asarray((1.0 - lam) * emp_dist + lam * uniform, dtype=np.float64)

    def _random_walk(self, start: Movement, lambda_t: float) -> Trajectory:
        self._ensure_cache_valid()
        movements = [start]
        curr_key = self._quantize_state(start.delta_state)
        visited = {curr_key}
        stop_reason = "max_random_walk_steps"
        from_vel, from_dir, from_tempo = start.velocity, start.direction, start.rhythm_signature.tempo_ratio

        for _ in range(self._config.max_random_walk_steps):
            cache = self._candidate_cache.get(curr_key)
            if cache is None:
                stop_reason = "dead_end"
                break
            if cache["n"] == 1:
                next_m = cache["movements"][0]
            else:
                post = self._theta.get_transition_probabilities(curr_key) if hasattr(self._theta, "get_transition_probabilities") else None
                weights = self._compute_transition_weights_cached(from_vel, from_dir, from_tempo, cache, post, lambda_t)
                idx = int(np.argmax(weights)) if lambda_t < 0.1 else int(np.random.choice(cache["n"], p=weights))
                next_m = cache["movements"][idx]

            movements.append(next_m)
            next_key = self._quantize_state(next_m.delta_state)
            if next_key in visited:
                stop_reason = "cycle"
                break
            if len(movements) >= self._config.max_trajectory_len:
                stop_reason = "max_length"
                break
            visited.add(next_key)
            curr_key = next_key
            from_vel, from_dir, from_tempo = next_m.velocity, next_m.direction, next_m.rhythm_signature.tempo_ratio

        truncated = len(movements) < self._config.min_trajectory_len or len(movements) < self._config.max_trajectory_len
        return Trajectory(
            movements=tuple(movements),
            coherence_score=0.0,
            invalidation_step=None,
            terminal_state=TerminalState(state_vector=movements[-1].delta_state, step_index=len(movements) - 1, confidence=0.0),
            metadata={"truncated": truncated, "stop_reason": stop_reason},
        )

    def generate_candidates(self, start_movement: Movement, lambda_t: float, num_candidates: int) -> list[Trajectory]:
        self._ensure_cache_valid()
        if lambda_t < 0.1:
            traj = self._random_walk(start_movement, lambda_t=0.0)
            return [traj] * max(num_candidates, 1) if (1 <= len(traj.movements) <= self._config.max_trajectory_len) else []

        candidates = [self._random_walk(start_movement, lambda_t) for _ in range(num_candidates)]
        valid_min = [t for t in candidates if self._config.min_trajectory_len <= len(t.movements) <= self._config.max_trajectory_len]
        return valid_min[:num_candidates] if valid_min else [t for t in candidates if 1 <= len(t.movements) <= self._config.max_trajectory_len][:num_candidates]

    def compute_transition_weights(self, start_movement: Movement, successors: list[Movement], lambda_t: float, posterior: Optional[dict] = None) -> np.ndarray:
        cache = {
            "movements": np.array(successors, dtype=object),
            "velocities": np.array([m.velocity for m in successors], dtype=np.float64),
            "directions": np.array([m.direction for m in successors], dtype=np.float64),
            "tempos": np.array([m.rhythm_signature.tempo_ratio for m in successors], dtype=np.float64),
            "delta_states": np.array([m.delta_state for m in successors], dtype=np.float64),
            "quantized_keys": tuple(self._quantize_state(m.delta_state) for m in successors),
            "n": len(successors),
        }
        return self._compute_transition_weights_cached(start_movement.velocity, start_movement.direction, start_movement.rhythm_signature.tempo_ratio, cache, posterior, lambda_t)