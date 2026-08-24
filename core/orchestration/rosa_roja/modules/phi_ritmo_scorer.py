"""PhiRitmoScorer: Computes Φ_Ritmo coherence scores and invalidation steps."""

from __future__ import annotations

import math
from typing import Optional
import numpy as np

from ..domain.trajectory import Trajectory, TerminalState


class PhiRitmoScorer:
    """Computes Φ_Ritmo trajectory coherence scores and invalidation points."""
    
    def __init__(
        self,
        rhythm_weight: float = 0.5,
        invalidation_threshold: float = 0.5,
    ):
        self.rhythm_weight = rhythm_weight
        self.invalidation_threshold = invalidation_threshold
    
    def score_trajectory(
        self,
        traj: Trajectory,
        lambda_t: float,
        entropy: float,
    ) -> Trajectory:
        """Compute Φ_Ritmo score and invalidation step for a trajectory."""
        length = len(traj.movements)
        
        # ΔH = information gain from trajectory (entropy reduction)
        delta_h = entropy / (length + 1e-6)
        
        # Vectorized rhythm coherence computation
        velocities = traj.velocities
        directions = traj.directions
        
        # Velocity consistency
        vel_consistency = 1.0 - np.std(velocities) / (np.mean(np.abs(velocities)) + 1e-6)
        vel_consistency = max(0.0, min(1.0, vel_consistency))
        
        # Direction consistency
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
        
        # ρ(T) = rhythm_weight * (0.4 * vel + 0.4 * dir + 0.2 * tempo)
        rho = self.rhythm_weight * (0.4 * vel_consistency + 0.4 * dir_consistency + 0.2 * tempo_consistency)
        
        # Φ_Ritmo(T) = (λ_t * ΔH + ρ) / |T|
        phi = (lambda_t * delta_h + rho) / length
        
        # Compute invalidation step
        invalidation_step = self._find_invalidation_step_vectorized(traj, phi)
        
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
    
    def _find_invalidation_step_vectorized(
        self,
        traj: Trajectory,
        initial_phi: float,
    ) -> Optional[int]:
        """Prefix-scan invalidation detection using running accumulators.

        Equivalent to evaluating the prefix Φ_Ritmo for every prefix length,
        but O(L) with scalar math instead of O(L²) numpy reductions.
        """
        movements = traj.movements
        length = len(movements)
        if length < 2:
            return None

        min_phi_threshold = initial_phi * self.invalidation_threshold

        directions = traj.directions
        # Consecutive direction dot products; prefix mean over first k dots.
        dots = np.sum(directions[1:] * directions[:-1], axis=1)
        dot_cum = np.cumsum(dots)

        velocities = traj.velocities.tolist()
        tempos = [m.rhythm_signature.tempo_ratio for m in movements]

        rw = self.rhythm_weight
        v_sum = v_abs = v_sq = float(velocities[0])
        t_sum = t_abs = t_sq = float(tempos[0])

        for k in range(1, length):
            n = k + 1
            v = velocities[k]
            t = tempos[k]
            av = abs(v)
            at = abs(t)
            v_sum += v; v_abs += av; v_sq += v * v
            t_sum += t; t_abs += at; t_sq += t * t

            v_mean = v_sum / n
            t_mean = t_sum / n
            v_std = math.sqrt(max(0.0, v_sq / n - v_mean * v_mean))
            t_std = math.sqrt(max(0.0, t_sq / n - t_mean * t_mean))

            vel_consistency = 1.0 - v_std / (v_abs / n + 1e-6)
            vel_consistency = max(0.0, min(1.0, vel_consistency))

            dir_consistency = max(0.0, float(dot_cum[k - 1]) / k)

            tempo_consistency = 1.0 - t_std / (t_abs / n + 1e-6)
            tempo_consistency = max(0.0, min(1.0, tempo_consistency))

            prefix_phi = rw * (
                0.4 * vel_consistency + 0.4 * dir_consistency + 0.2 * tempo_consistency
            ) / n

            if prefix_phi < min_phi_threshold:
                return k

        return None