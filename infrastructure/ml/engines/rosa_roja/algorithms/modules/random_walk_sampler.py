"""RandomWalkSampler: Continuous phase-space candidate trajectory generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from domain.entities.rosa_roja.movement import Movement
from domain.entities.rosa_roja.trajectory import Trajectory, TerminalState
from domain.entities.rosa_roja.theta_belief import StateKey
from domain.ports.rosa_roja.guided_field import GuidedFieldPort


DEFAULT_MAX_STEPS: int = 100
DEFAULT_MIN_TRAJECTORY_LEN: int = 11
DEFAULT_MAX_TRAJECTORY_LEN: int = 15
DEFAULT_QUANTIZATION_DECIMALS: int = 2
DEFAULT_RBF_BANDWIDTH: float = 0.5
DEFAULT_LAMBDA_DETERMINISTIC_THRESHOLD: float = 0.1
DEFAULT_VELOCITY_EPSILON: float = 1e-6
DEFAULT_TEMPO_EPSILON: float = 1e-6
DEFAULT_WEIGHT_VELOCITY: float = 0.4
DEFAULT_WEIGHT_DIRECTION: float = 0.4
DEFAULT_WEIGHT_TEMPO: float = 0.2
DEFAULT_TEMPO_SIMILARITY: float = 0.5


@dataclass
class RandomWalkConfig:
    max_random_walk_steps: int = DEFAULT_MAX_STEPS
    min_trajectory_len: int = DEFAULT_MIN_TRAJECTORY_LEN
    max_trajectory_len: int = DEFAULT_MAX_TRAJECTORY_LEN
    quantization_decimals: int = DEFAULT_QUANTIZATION_DECIMALS
    rbf_bandwidth: float = DEFAULT_RBF_BANDWIDTH
    deterministic_lambda_threshold: float = DEFAULT_LAMBDA_DETERMINISTIC_THRESHOLD
    velocity_epsilon: float = DEFAULT_VELOCITY_EPSILON
    tempo_epsilon: float = DEFAULT_TEMPO_EPSILON
    weight_velocity: float = DEFAULT_WEIGHT_VELOCITY
    weight_direction: float = DEFAULT_WEIGHT_DIRECTION
    weight_tempo: float = DEFAULT_WEIGHT_TEMPO
    default_tempo_similarity: float = DEFAULT_TEMPO_SIMILARITY


class RandomWalkSampler:
    """Generates trajectories along manifold flow without fabricating phase-shifted transitions."""

    def __init__(
        self,
        config: RandomWalkConfig,
        transition_graph: dict[StateKey, list[Movement]],
        theta_belief: Any,
        quantize_state_func: Any,
        guided_field: Optional[GuidedFieldPort] = None,
    ) -> None:
        self._config = config
        self._transition_graph = transition_graph
        self._theta = theta_belief
        self._quantize_state = quantize_state_func
        self._guided_field = guided_field
        self._graph_version = 0
        self._cache_version = -1
        self._candidate_cache: dict[Any, dict[str, Any]] = {}
        self._precompute_candidate_data()

    def set_guided_field(self, guided_field: Optional[GuidedFieldPort]) -> None:
        """Attach or update the GuidedFieldPort provider."""
        self._guided_field = guided_field

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
        cfg = self._config
        vel_sim = np.maximum(0.0, 1.0 - np.abs(cache["velocities"] - from_vel) / (abs(from_vel) + cfg.velocity_epsilon))
        dir_sim = np.maximum(0.0, cache["directions"] @ from_dir)
        tempo_sim = np.where(
            (from_tempo > cfg.tempo_epsilon) & (cache["tempos"] > cfg.tempo_epsilon),
            1.0 - np.minimum(1.0, np.abs(np.log(cache["tempos"] / from_tempo))),
            cfg.default_tempo_similarity,
        )
        coherence = np.maximum(0.0, cfg.weight_velocity * vel_sim + cfg.weight_direction * dir_sim + cfg.weight_tempo * tempo_sim)
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
            if cache is None and curr_key in self._transition_graph:
                self._precompute_candidate_data()
                cache = self._candidate_cache.get(curr_key)
            if cache is None:
                if self._guided_field is None:
                    stop_reason = "dead_end"
                    break

                # Guided Importance Sampling: query macro gradient and prior distribution
                macro_dir, conf = self._guided_field.get_macro_gradient(movements[-1].timestamp)
                motif_priors = self._guided_field.get_motif_prior("default")

                norm_macro = float(np.linalg.norm(macro_dir))
                if norm_macro <= 1e-6 and conf <= 0.0 and not motif_priors:
                    stop_reason = "dead_end"
                    break

                # Synthesize next state along joint alignment (Fourier trend + kinematic momentum)
                if norm_macro > 1e-6:
                    unit_macro = macro_dir / norm_macro
                    if unit_macro.shape != from_dir.shape:
                        if unit_macro.size == 1 and from_dir.size > 0:
                            unit_macro = np.sign(float(unit_macro[0])) * from_dir
                        else:
                            unit_macro = from_dir
                    alpha = float(np.clip(conf, 0.2, 0.8))
                    blended = (1.0 - alpha) * from_dir + alpha * unit_macro
                    norm_b = float(np.linalg.norm(blended))
                    eff_dir = blended / norm_b if norm_b > 1e-6 else from_dir
                else:
                    eff_dir = from_dir

                dt = start.delta_time if start.delta_time > 0 else 1.0
                mag = max(float(from_vel) * dt, 1e-4) if from_vel > 0 else (float(np.linalg.norm(start.delta_state)) or 1e-3)
                synth_delta = eff_dir * mag
                next_m = Movement.from_raw(
                    delta_state=synth_delta,
                    delta_time=dt,
                    timestamp=movements[-1].timestamp + dt,
                    prev_movement=movements[-1],
                )
            elif cache["n"] == 1:
                next_m = cache["movements"][0]
            else:
                post = self._theta.get_transition_probabilities(curr_key) if hasattr(self._theta, "get_transition_probabilities") else None
                weights = self._compute_transition_weights_cached(from_vel, from_dir, from_tempo, cache, post, lambda_t)
                idx = int(np.argmax(weights)) if lambda_t < self._config.deterministic_lambda_threshold else int(np.random.choice(cache["n"], p=weights))
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
        if lambda_t < self._config.deterministic_lambda_threshold:
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