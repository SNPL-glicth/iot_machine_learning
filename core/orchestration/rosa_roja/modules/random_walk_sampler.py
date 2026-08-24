"""RandomWalkSampler: Generates candidate trajectories via λ-controlled random walk.

Optimizations:
- Pre-computes candidate arrays for vectorized operations
- Batched random walk generation using NumPy
- Early stopping when lambda_t -> 0 (exploitation mode)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
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


class RandomWalkSampler:
    """Generates trajectories via λ-controlled random walk on transition graph.
    
    Optimizations:
    - Pre-computes candidate arrays for vectorized operations
    - Batched random walk generation using NumPy
    - Early stopping when lambda_t -> 0 (exploitation mode)
    """
    
    def __init__(
        self,
        config: RandomWalkConfig,
        transition_graph: dict[StateKey, list[Movement]],
        theta_belief,
        quantize_state_func,
    ):
        self._config = config
        self._transition_graph = transition_graph
        self._theta = theta_belief
        self._quantize_state = quantize_state_func
        
        # Graph version for cache invalidation
        self._graph_version = 0
        self._cache_version = -1

        # Pre-compute candidate data arrays for fast access
        self._precompute_candidate_data()

    def update_transition_graph(self, transition_graph: dict[StateKey, list[Movement]]) -> None:
        """Update the transition graph and rebuild candidate cache."""
        self._transition_graph = transition_graph
        self._graph_version += 1
        self._precompute_candidate_data()

    def _ensure_cache_valid(self):
        """Ensure candidate cache is valid for current graph."""
        if self._cache_version != self._graph_version:
            self._cache_version = self._graph_version
            self._precompute_candidate_data()
    
    def _precompute_candidate_data(self):
        """Pre-compute candidate arrays for fast vectorized access."""
        self._candidate_cache = {}
        for state_key, movements in self._transition_graph.items():
            n = len(movements)
            if n == 0:
                continue
            # Pre-extract arrays for fast access
            velocities = np.array([m.velocity for m in movements], dtype=np.float64)
            directions = np.array([m.direction for m in movements], dtype=np.float64)
            tempos = np.array([m.rhythm_signature.tempo_ratio for m in movements], dtype=np.float64)
            delta_states = np.array([m.delta_state for m in movements], dtype=np.float64)
            quantized_keys = tuple(self._quantize_state(m.delta_state) for m in movements)
            movements_array = np.array(movements, dtype=object)
            
            self._candidate_cache[state_key] = {
                'movements': movements_array,
                'velocities': velocities,
                'directions': directions,
                'tempos': tempos,
                'delta_states': delta_states,
                'quantized_keys': quantized_keys,
                'n': n,
            }
    
    def generate_candidates(
        self,
        start_movement: Movement,
        lambda_t: float,
        num_candidates: int,
    ) -> list[Trajectory]:
        """Generate multiple candidate trajectories using batched walks."""
        self._ensure_cache_valid()
        if lambda_t < 0.1:
            # Exploitation mode: greedy selection with weights shared across
            # walks of this generation call.
            shared_weights: dict = {}
            return self._generate_greedy_candidates(start_movement, shared_weights, num_candidates)

        # Batched walk generation
        return self._generate_batched_walks(start_movement, lambda_t, num_candidates)

    def _generate_greedy_candidates(
        self,
        start_movement: Movement,
        weight_cache: dict,
        num_candidates: int = 0,
    ) -> list[Trajectory]:
        """Fast path: greedy selection for low lambda (exploitation).

        The greedy walk is deterministic (argmax over fixed weights), so a
        single walk replicated `num_candidates` times is equivalent to
        running it repeatedly, at a fraction of the cost.
        """
        self._ensure_cache_valid()
        traj = self._greedy_walk(start_movement, weight_cache)
        if not (1 <= len(traj.movements) <= self._config.max_trajectory_len):
            return []
        return [traj] * max(num_candidates, 1)
    
    def _greedy_walk(self, start: Movement, weight_cache: Optional[dict] = None) -> Trajectory:
        """Single greedy walk (always pick highest weight)."""
        self._ensure_cache_valid()
        movements = [start]
        current_state_key = self._quantize_state(start.delta_state)
        visited_states = {current_state_key}
        zero_velocity_count = 0
        stop_reason = "max_random_walk_steps"

        if weight_cache is None:
            weight_cache = {}

        for step in range(self._config.max_random_walk_steps):
            cache = self._candidate_cache.get(current_state_key)
            if cache is None:
                stop_reason = "dead_end"
                break
            
            n_candidates = cache['n']
            
            # Greedy: pick best candidate based on coherence
            prev = movements[-1]
            
            # Use shared weights if possible (keyed by predecessor + state)
            cache_key = (id(prev), current_state_key)
            if cache_key not in weight_cache:
                posterior = self._theta.get_transition_probabilities(current_state_key)
                weights = self._compute_transition_weights_cached(
                    prev.velocity, prev.direction,
                    prev.rhythm_signature.tempo_ratio,
                    cache, posterior, 0.0
                )
                weight_cache[cache_key] = weights
            else:
                weights = weight_cache[cache_key]
            
            idx = int(np.argmax(weights))
            next_m = cache['movements'][idx]
            
            # Track zero-velocity
            if next_m.velocity < 1e-6:
                zero_velocity_count += 1
                if zero_velocity_count > 3:
                    stop_reason = "degenerate_zero_velocity"
                    break
            else:
                zero_velocity_count = 0
            
            movements.append(next_m)
            next_state_key = self._quantize_state(next_m.delta_state)
            
            if next_state_key in visited_states:
                stop_reason = "cycle"
                break
            visited_states.add(next_state_key)
            
            current_state_key = next_state_key
            
            if len(movements) >= self._config.max_trajectory_len:
                stop_reason = "max_length"
                break
        
        truncated = len(movements) < self._config.max_trajectory_len
        return Trajectory(
            movements=tuple(movements),
            coherence_score=0.0,
            invalidation_step=None,
            terminal_state=TerminalState(
                state_vector=movements[-1].delta_state,
                step_index=len(movements) - 1,
                confidence=0.0,
            ),
            metadata={
                "truncated": truncated,
                "stop_reason": stop_reason,
            },
        )
    
    # Backward compatibility
    def _random_walk(self, start: Movement, lambda_t: float) -> Trajectory:
        """Backward compatibility: delegate to greedy walk for lambda=0, batched for others."""
        self._ensure_cache_valid()
        if lambda_t < 0.1:
            return self._greedy_walk(start)
        return self._batched_random_walk(start, lambda_t, 1)[0]
    
    def _generate_batched_walks(
        self,
        start_movement: Movement,
        lambda_t: float,
        num_candidates: int,
    ) -> list[Trajectory]:
        """Generate multiple walks using batched vectorized operations."""
        self._ensure_cache_valid()
        candidates = []
        
        # Process in batches of 32 for better vectorization
        batch_size = min(32, num_candidates)
        n_batches = (num_candidates + batch_size - 1) // batch_size
        
        for _ in range(n_batches):
            current_batch = min(batch_size, num_candidates - len(candidates))
            
            # Run parallel walks in batch
            batch_results = self._batched_random_walk(start_movement, lambda_t, current_batch)
            candidates.extend(batch_results)
        
        # Filter by length - enforce min_trajectory_len, but fall back to shorter trajectories
        # if no candidates meet the minimum (e.g., all hit dead ends/cycles)
        valid_min = [t for t in candidates 
                     if self._config.min_trajectory_len <= len(t.movements) <= self._config.max_trajectory_len]
        if valid_min:
            valid = valid_min
        else:
            # Fallback: allow shorter trajectories if no candidates meet min_trajectory_len
            valid = [t for t in candidates 
                     if len(t.movements) >= 1 and len(t.movements) <= self._config.max_trajectory_len]
        return valid[:num_candidates]
    
    def _batched_random_walk(
        self,
        start: Movement,
        lambda_t: float,
        batch_size: int,
    ) -> list[Trajectory]:
        """Batched random walk with optimized weight caching."""
        batch_movements = [[start] for _ in range(batch_size)]
        batch_state_keys = [self._quantize_state(start.delta_state) for _ in range(batch_size)]
        batch_visited = [{k} for k in batch_state_keys]
        batch_zero_vel = [0] * batch_size
        batch_stop_reasons = ["max_random_walk_steps"] * batch_size
        batch_active = [True] * batch_size
        
        # Pre-fetch start movement data
        from_vel = start.velocity
        from_dir = start.direction
        from_tempo = start.rhythm_signature.tempo_ratio
        
        # Weight cache for avoiding recomputation
        weight_cache = {}
        
        for step in range(self._config.max_random_walk_steps):
            if not any(batch_active):
                break
            
            for i in range(batch_size):
                if not batch_active[i]:
                    continue
                
                cache = self._candidate_cache.get(batch_state_keys[i])
                if cache is None:
                    # Honest dead end: no observed transition from this state.
                    # Fabricating a continuation from a "similar" state would
                    # produce phase-shifted predictions and spurious deviations.
                    batch_stop_reasons[i] = "dead_end"
                    batch_active[i] = False
                    continue
                
                n_candidates = cache['n']
                
                if n_candidates == 1:
                    next_m = cache['movements'][0]
                else:
                    # Use cached weights if possible
                    cache_key = (batch_state_keys[i], lambda_t)
                    if cache_key not in weight_cache:
                        posterior = self._theta.get_transition_probabilities(batch_state_keys[i])
                        weights = self._compute_transition_weights_cached(
                            from_vel, from_dir, from_tempo,
                            cache, posterior, lambda_t
                        )
                        weight_cache[cache_key] = weights
                    else:
                        weights = weight_cache[cache_key]
                    
                    idx = np.random.choice(n_candidates, p=weights)
                    next_m = cache['movements'][idx]
                
                if next_m.velocity < 1e-6:
                    batch_zero_vel[i] += 1
                    if batch_zero_vel[i] > 3:
                        batch_stop_reasons[i] = "degenerate_zero_velocity"
                        batch_active[i] = False
                        continue
                else:
                    batch_zero_vel[i] = 0
                
                batch_movements[i].append(next_m)
                next_state_key = self._quantize_state(next_m.delta_state)
                
                if next_state_key in batch_visited[i]:
                    batch_stop_reasons[i] = "cycle"
                    batch_active[i] = False
                    continue
                
                batch_visited[i].add(next_state_key)
                batch_state_keys[i] = next_state_key
                
                from_vel = next_m.velocity
                from_dir = next_m.direction
                from_tempo = next_m.rhythm_signature.tempo_ratio
                
                if len(batch_movements[i]) >= self._config.max_trajectory_len:
                    batch_stop_reasons[i] = "max_length"
                    batch_active[i] = False
        
        # Build trajectory objects
        results = []
        for i in range(batch_size):
            if len(batch_movements[i]) < 1:
                continue
            
            truncated = len(batch_movements[i]) < self._config.min_trajectory_len or \
                       len(batch_movements[i]) < self._config.max_trajectory_len
            traj = Trajectory(
                movements=tuple(batch_movements[i]),
                coherence_score=0.0,
                invalidation_step=None,
                terminal_state=TerminalState(
                    state_vector=batch_movements[i][-1].delta_state,
                    step_index=len(batch_movements[i]) - 1,
                    confidence=0.0,
                ),
                metadata={
                    "truncated": truncated,
                    "stop_reason": batch_stop_reasons[i],
                },
            )
            results.append(traj)
        
        return results
    
    def _compute_transition_weights_cached(
        self,
        from_vel: float,
        from_dir: np.ndarray,
        from_tempo: float,
        cache: dict,
        posterior: Optional[dict],
        lambda_t: float,
    ) -> np.ndarray:
        """Fast weight computation using pre-cached arrays."""
        n = cache['n']
        if n == 1:
            return np.array([1.0])
        
        uniform = np.full(n, 1.0 / n)
        
        # Vectorized coherence using cached arrays
        vel_sim = 1.0 - np.abs(cache['velocities'] - from_vel) / (np.abs(from_vel) + 1e-6)
        dir_sim = np.maximum(0.0, cache['directions'] @ from_dir)
        tempo_sim = np.where(
            (from_tempo > 1e-6) & (cache['tempos'] > 1e-6),
            1.0 - np.minimum(1.0, np.abs(np.log(cache['tempos'] / from_tempo))),
            0.5
        )
        coherence = np.maximum(0.0, 0.4 * vel_sim + 0.4 * dir_sim + 0.2 * tempo_sim)
        coherence_dist = coherence / coherence.sum() if coherence.sum() > 0 else uniform.copy()
        
        if posterior:
            prior = np.array([posterior.get(key, 0.0) for key in cache['quantized_keys']])
            prior_dist = prior / prior.sum() if prior.sum() > 0 else uniform.copy()
        else:
            prior_dist = uniform.copy()
        
        empirical = prior_dist * coherence_dist
        empirical_sum = empirical.sum()
        if empirical_sum <= 0:
            empirical = uniform.copy()
        else:
            empirical = empirical / empirical_sum
        
        lam = float(np.clip(lambda_t, 0.0, 1.0))
        return (1.0 - lam) * empirical + lam * uniform

    # Backward compatibility
    def compute_transition_weights(self, start_movement, successors, lambda_t, posterior=None):
        """Backward compatibility wrapper for old signature."""
        from_vel = start_movement.velocity
        from_dir = start_movement.direction
        from_tempo = start_movement.rhythm_signature.tempo_ratio
        
        # Build cache from successors
        n = len(successors)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])
        
        cache = {
            'n': n,
            'movements': np.array(successors, dtype=object),
            'velocities': np.array([m.velocity for m in successors], dtype=np.float64),
            'directions': np.array([m.direction for m in successors], dtype=np.float64),
            'tempos': np.array([m.rhythm_signature.tempo_ratio for m in successors], dtype=np.float64),
            'delta_states': np.array([m.delta_state for m in successors], dtype=np.float64),
            'quantized_keys': tuple(self._quantize_state(m.delta_state) for m in successors),
        }
        
        return self._compute_transition_weights_cached(
            from_vel, from_dir, from_tempo, cache, posterior, lambda_t
        )