"""Module 2: Trajectory & Rhythm Density Generator (Optimized)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import random

# Try to import Numba for JIT compilation
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback decorator that does nothing
    def jit(nopython=True, cache=False):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False

from ..domain.movement import Movement
from ..domain.trajectory import Trajectory, TerminalState
from ..domain.theta_belief import StateKey, ThetaBelief


@dataclass
class RhythmTrajectoryGenerator:
    """
    Trajectory & Rhythm Density Generator.
    
    Generates Top-K candidate trajectories T (11 ≤ |T| ≤ 15).
    Normalizes information gain by length |T| to eliminate long-trajectory bias.
    Dynamically adjusts exploration factor λ_t = min(H(Θ|D_t)/H_max, 1 - DriftScore_t).
    
    λ_t modulates BOTH scoring and generation: it interpolates the random-walk
    transition sampling between exploitation of observed evidence (λ→0) and
    uniform exploration of the transition graph (λ→1).
    
    H(Θ|D_t) comes from an explicit decaying posterior (ThetaBelief) over
    transition modes; walk sampling weights are the posterior transition
    probabilities modulated by rhythm coherence.
    
    Equation: Φ_Ritmo(T) = 1/|T| [λ_t · ΔH(T|D_t) + ρ(T)]
    """
    
    min_trajectory_len: int = 11
    max_trajectory_len: int = 15
    top_k: int = 5
    rhythm_weight: float = 0.5           # γ in ρ(T)
    max_entropy: float = 1.0             # H_max for λ_t normalization (ThetaBelief entropy is already in [0, 1])
    oversample_factor: int = 3           # Generate top_k * oversample_factor candidates
    max_random_walk_steps: int = 100     # Guard against infinite loops
    invalidation_threshold: float = 0.5  # Φ_Ritmo decay ratio for invalidation
    theta_alpha: float = 0.95            # Exponential forgetting factor for ThetaBelief
    quantization_decimals: int = 2       # Decimal places for state key quantization
    
    def __post_init__(self):
        self._transition_graph: dict[tuple, list[Movement]] = {}
        self._history: list[Movement] = []
        self._theta = ThetaBelief(alpha=self.theta_alpha)
        self._latest_state_key: Optional[StateKey] = None
        self._symbol: Optional[str] = None  # For per-symbol isolation
        self._exploration_boost: int = 0  # Remaining events with forced λ=1.0
    
    def _quantize_state(self, state: np.ndarray) -> StateKey:
        """Quantize state vector to configured decimal precision for graph/belief keys."""
        return tuple(np.round(state, self.quantization_decimals))
    
    def set_symbol(self, symbol: str) -> None:
        """Set current symbol context for per-symbol state isolation."""
        if self._symbol != symbol:
            self.reset()
            self._symbol = symbol
    
    def reset(self) -> None:
        """Reset internal state (clears history, transition graph and belief)."""
        self._transition_graph.clear()
        self._history.clear()
        self._theta.reset()
        self._latest_state_key = None
        self._exploration_boost = 0
    
    def boost_exploration(self, events: int) -> None:
        """Force λ_t = 1.0 for the next `events` generations (max exploration)."""
        self._exploration_boost = max(0, events)
    
    def generate_candidate_trajectories(
        self, 
        latest_movement: Movement, 
        drift_score: float
    ) -> list[Trajectory]:
        """
        Generate Top-K candidate trajectories from current state.
        
        Args:
            latest_movement: The most recent validated movement from Module 1
            drift_score: Normalized DriftScore_t ∈ [0.0, 1.0] from drift sensors
            
        Returns:
            List of Trajectory objects sorted by Φ_Ritmo score (descending)
        """
        # Update internal history and transition graph
        self._history.append(latest_movement)
        if len(self._history) > 200:  # Cap history for graph
            self._history.pop(0)
        self._update_transition_graph()
        self._latest_state_key = self._quantize_state(latest_movement.delta_state)
        self._update_belief()
        
        if len(self._history) < self.min_trajectory_len:
            return []
        
        # Compute exploration factor λ_t
        entropy = self._compute_entropy()
        lambda_t = self._compute_lambda(entropy, drift_score)
        
        # Exploration boost forces maximum exploration for N events
        if self._exploration_boost > 0:
            lambda_t = 1.0
            self._exploration_boost -= 1
        
        # Generate candidate trajectories via controlled random walk
        candidates = self._generate_candidates(latest_movement, lambda_t)
        
        # Score and select top-K
        scored = []
        for traj in candidates:
            scored_traj = self._phi_ritmo(traj, lambda_t, entropy)
            scored.append((scored_traj.coherence_score, scored_traj))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:self.top_k]]
    
    def _compute_lambda(self, entropy: float, drift_score: float) -> float:
        """λ_t = min(H(Θ|D_t)/H_max, 1 - DriftScore_t)"""
        normalized_entropy = min(entropy / self.max_entropy, 1.0) if self.max_entropy > 0 else 0.0
        drift_penalty = max(0.0, 1.0 - drift_score)
        return min(normalized_entropy, drift_penalty)
    
    def _compute_entropy(self) -> float:
        """
        Conditional entropy H(Θ|D_t) ∈ [0, 1], delegated to the explicit
        posterior. Local entropy at the current state when known; global
        average otherwise.
        """
        return self._theta.compute_entropy(self._latest_state_key)
    
    def _update_belief(self) -> None:
        """
        Feed observed transitions into the ThetaBelief posterior.
        
        On first call, replays the full history; afterwards feeds only the
        newest transition incrementally (the belief itself holds the
        decaying long-term memory).
        """
        if len(self._history) < 2:
            return
        
        if self._theta.total_updates == 0:
            for curr, nxt in zip(self._history, self._history[1:]):
                from_key = self._quantize_state(curr.delta_state)
                to_key = self._quantize_state(nxt.delta_state)
                self._theta.update(from_key, to_key)
        else:
            prev = self._history[-2]
            latest = self._history[-1]
            self._theta.update(
                self._quantize_state(prev.delta_state),
                self._quantize_state(latest.delta_state),
            )
    
    def _update_transition_graph(self) -> None:
        """Build/update transition graph from movement history."""
        self._transition_graph = {}
        for i in range(len(self._history) - 1):
            curr = self._history[i]
            next_m = self._history[i + 1]
            
            # Discretize state for graph key (quantize delta_state)
            state_key = self._quantize_state(curr.delta_state)
            
            if state_key not in self._transition_graph:
                self._transition_graph[state_key] = []
            self._transition_graph[state_key].append(next_m)
    
    def _generate_candidates(self, start_movement: Movement, lambda_t: float) -> list[Trajectory]:
        """Generate candidate trajectories via λ-controlled random walk on transition graph."""
        candidates = []
        num_candidates = self.top_k * self.oversample_factor
        
        for _ in range(num_candidates):
            traj = self._random_walk(start_movement, lambda_t)
            if self.min_trajectory_len <= len(traj.movements) <= self.max_trajectory_len:
                candidates.append(traj)
        
        return candidates
    
    def _random_walk(self, start: Movement, lambda_t: float = 0.5) -> Trajectory:
        """
        Generate single trajectory via λ-controlled random walk on transition graph.
        
        The walk stops cleanly when evidence runs out (dead end, cycle,
        degenerate zero-velocity steps) instead of padding with repeated
        movements. Truncation metadata is exposed via trajectory.metadata.
        
        Includes cycle detection and max steps guard to prevent infinite loops.
        """
        movements = [start]
        current_state_key = self._quantize_state(start.delta_state)
        visited_states = {current_state_key}
        zero_velocity_count = 0
        stop_reason = "max_random_walk_steps"
        
        # Pre-compute uniform for fallback
        for step in range(self.max_random_walk_steps):
            next_movements = self._transition_graph.get(current_state_key)
            if not next_movements:
                stop_reason = "dead_end"
                break
            
            n_candidates = len(next_movements)
            
            # Fast path: single candidate
            if n_candidates == 1:
                next_m = next_movements[0]
            else:
                # λ-controlled sampling with vectorized weight computation
                posterior = self._theta.get_transition_probabilities(current_state_key)
                weights = self._compute_transition_weights_vectorized(
                    movements[-1], next_movements, lambda_t, posterior=posterior
                )
                # Use numpy random choice for speed
                idx = np.random.choice(n_candidates, p=weights)
                next_m = next_movements[idx]
            
            # Track zero-velocity steps (degenerate)
            if next_m.velocity < 1e-6:
                zero_velocity_count += 1
                if zero_velocity_count > 3:
                    stop_reason = "degenerate_zero_velocity"
                    break
            else:
                zero_velocity_count = 0
            
            movements.append(next_m)
            next_state_key = self._quantize_state(next_m.delta_state)
            
            # Cycle detection: if we revisit a state, break
            if next_state_key in visited_states:
                stop_reason = "cycle"
                break
            visited_states.add(next_state_key)
            
            current_state_key = next_state_key
            
            # Stop if we've reached max trajectory length
            if len(movements) >= self.max_trajectory_len:
                stop_reason = "max_length"
                break
        
        truncated = len(movements) < self.max_trajectory_len
        
        return Trajectory(
            movements=tuple(movements),
            coherence_score=0.0,  # Will be computed by _phi_ritmo
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
    
    def _compute_transition_weights(self, from_movement: Movement, 
                                     candidates: list[Movement],
                                     lambda_t: float = 0.5,
                                     posterior: Optional[dict] = None) -> np.ndarray:
        """
        Compute λ-interpolated transition weights (original scalar version for compatibility).
        """
        return self._compute_transition_weights_vectorized(from_movement, candidates, lambda_t, posterior)
    
    def _compute_transition_weights_vectorized(self, from_movement: Movement, 
                                                candidates: list[Movement],
                                                lambda_t: float = 0.5,
                                                posterior: Optional[dict] = None) -> np.ndarray:
        """
        Vectorized λ-interpolated transition weights computation.
        
        Base distribution = posterior transition probabilities P(·|S_t, Θ_t)
        modulated by rhythm coherence (product of both), blended with a
        uniform distribution via λ: w = (1-λ)·posterior·coherence + λ·uniform.
        High entropy/low drift (λ→1) drives exploration; confident regimes
        (λ→0) exploit the belief. Transitions unseen by Θ get zero posterior
        mass and are only reachable through the exploration component.
        """
        n = len(candidates)
        if n == 1:
            return np.array([1.0])
        
        uniform = np.full(n, 1.0 / n)
        
        # Vectorized coherence computation
        cand_velocities = np.array([c.velocity for c in candidates])
        cand_directions = np.array([c.direction for c in candidates])
        cand_tempos = np.array([c.rhythm_signature.tempo_ratio for c in candidates])
        
        from_vel = from_movement.velocity
        from_dir = from_movement.direction
        from_tempo = from_movement.rhythm_signature.tempo_ratio
        
        # Velocity similarity (vectorized)
        vel_sim = 1.0 - np.abs(cand_velocities - from_vel) / (np.abs(from_vel) + 1e-6)
        
        # Direction similarity (vectorized dot product)
        dir_sim = np.maximum(0.0, np.dot(cand_directions, from_dir))
        
        # Tempo similarity (vectorized)
        tempo_sim = np.where(
            (from_tempo > 1e-6) & (cand_tempos > 1e-6),
            1.0 - np.minimum(1.0, np.abs(np.log(cand_tempos / from_tempo))),
            0.5
        )
        
        # Combine: 0.4 velocity + 0.4 direction + 0.2 tempo
        coherence = np.maximum(0.0, 0.4 * vel_sim + 0.4 * dir_sim + 0.2 * tempo_sim)
        
        coherence_dist = coherence / coherence.sum() if coherence.sum() > 0 else uniform.copy()
        
        if posterior:
            # Vectorized posterior lookup - convert to list of tuples for hashing
            cand_keys = [self._quantize_state(c.delta_state) for c in candidates]
            prior = np.array([posterior.get(key, 0.0) for key in cand_keys])
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
    
    def _phi_ritmo(self, traj: Trajectory, lambda_t: float, entropy: float) -> Trajectory:
        """Φ_Ritmo(T) = 1/|T| [λ_t · ΔH(T|D_t) + ρ(T)]
        
        Returns updated Trajectory with computed coherence_score and invalidation_step.
        """
        length = len(traj.movements)
        
        # ΔH = information gain from trajectory (simplified as entropy reduction)
        delta_h = entropy / (length + 1e-6)
        
        # Vectorized rhythm coherence computation
        velocities = traj.velocities
        directions = traj.directions
        
        # Velocity consistency
        vel_consistency = 1.0 - np.std(velocities) / (np.mean(np.abs(velocities)) + 1e-6)
        vel_consistency = max(0.0, min(1.0, vel_consistency))
        
        # Direction consistency (vectorized)
        if length > 1:
            dir_dots = np.sum(directions[1:] * directions[:-1], axis=1)
            dir_consistency = max(0.0, np.mean(dir_dots))
        else:
            dir_consistency = 0.0
        
        # Tempo consistency
        tempo_ratios = np.array([m.rhythm_signature.tempo_ratio for m in traj.movements])
        if length > 1:
            tempo_consistency = 1.0 - np.std(tempo_ratios) / (np.mean(np.abs(tempo_ratios)) + 1e-6)
            tempo_consistency = max(0.0, min(1.0, tempo_consistency))
        else:
            tempo_consistency = 0.0
        
        # Update formula: 0.4 vel + 0.4 dir + 0.2 tempo
        rho = self.rhythm_weight * (0.4 * vel_consistency + 0.4 * dir_consistency + 0.2 * tempo_consistency)
        
        # Normalize by trajectory length |T|
        phi = (lambda_t * delta_h + rho) / length
        
        # Compute invalidation step based on Φ_Ritmo decay
        invalidation_step = self._find_invalidation_step_vectorized(traj, phi)
        
        # Return updated trajectory with computed coherence
        return Trajectory(
            movements=traj.movements,
            coherence_score=phi,
            invalidation_step=invalidation_step,
            terminal_state=TerminalState(
                state_vector=traj.terminal_state.state_vector,
                step_index=traj.terminal_state.step_index,
                confidence=phi,
            ),
            metadata=traj.metadata,
        )
    
    def _find_invalidation_step_vectorized(self, traj: Trajectory, initial_phi: float) -> Optional[int]:
        """
        Vectorized invalidation step detection.
        
        Tracks cumulative Φ_Ritmo decay along the trajectory.
        Invalidation occurs when Φ_Ritmo(t) < invalidation_threshold * Φ_Ritmo(0).
        """
        if len(traj.movements) < 2:
            return None
        
        min_phi_threshold = initial_phi * self.invalidation_threshold
        movements = traj.movements
        length = len(movements)
        
        # Pre-extract arrays for vectorized prefix computation
        velocities = traj.velocities
        directions = traj.directions
        tempo_ratios = np.array([m.rhythm_signature.tempo_ratio for m in movements])
        
        # Check each prefix length
        for step_idx in range(1, length):
            prefix_len = step_idx + 1
            
            # Prefix slices
            prefix_vel = velocities[:prefix_len]
            prefix_dir = directions[:prefix_len]
            prefix_tempo = tempo_ratios[:prefix_len]
            
            # Velocity consistency
            vel_consistency = 1.0 - np.std(prefix_vel) / (np.mean(np.abs(prefix_vel)) + 1e-6)
            vel_consistency = max(0.0, min(1.0, vel_consistency))
            
            # Direction consistency
            if prefix_len > 1:
                dir_dots = np.sum(prefix_dir[1:] * prefix_dir[:-1], axis=1)
                dir_consistency = max(0.0, np.mean(dir_dots))
            else:
                dir_consistency = 0.0
            
            # Tempo consistency
            if prefix_len > 1:
                tempo_consistency = 1.0 - np.std(prefix_tempo) / (np.mean(np.abs(prefix_tempo)) + 1e-6)
                tempo_consistency = max(0.0, min(1.0, tempo_consistency))
            else:
                tempo_consistency = 0.0
            
            # Simplified Φ_Ritmo for prefix
            prefix_phi = self.rhythm_weight * (0.4 * vel_consistency + 0.4 * dir_consistency + 0.2 * tempo_consistency) / prefix_len
            
            if prefix_phi < min_phi_threshold:
                return step_idx
        
        return None
    
    # Keep original for backward compatibility
    def _find_invalidation_step(self, traj: Trajectory, initial_phi: float) -> Optional[int]:
        return self._find_invalidation_step_vectorized(traj, initial_phi)