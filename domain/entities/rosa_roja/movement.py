"""Movement and RhythmSignature domain models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True, slots=True)
class RhythmSignature:
    """Temporal ratios and velocity deltas across transitions."""
    tempo_ratio: float              # Δt_i / Δt_{i-1}
    velocity_delta: float           # v_i - v_{i-1}
    acceleration: float             # (v_i - v_{i-1}) / Δt_i
    phase_angle: float              # Direction change in radians
    entropy_rate: float             # Local Shannon entropy of movement sequence


@dataclass(frozen=True, slots=True)
class Movement:
    """Agnostic state transition S_t → S_{t+1}."""
    delta_state: np.ndarray         # ΔS (multidimensional)
    delta_time: float               # Δt
    velocity: float                 # |ΔS| / Δt
    direction: np.ndarray           # Normalized ΔS vector
    rhythm_signature: RhythmSignature
    mahalanobis_distance: float     # For Module 1 filtering
    timestamp: float

    @classmethod
    def from_raw(cls, delta_state: np.ndarray, delta_time: float, timestamp: float,
                 mahalanobis_dist: float = 0.0, prev_movement: Optional["Movement"] = None) -> "Movement":
        """Factory method to create Movement from raw transition data.

        Args:
            delta_state: State change vector ΔS
            delta_time: Time delta Δt
            timestamp: Event timestamp
            mahalanobis_dist: Mahalanobis distance from Module 1
            prev_movement: Previous movement for rhythm computation
        """
        velocity = float(np.linalg.norm(delta_state)) / delta_time if delta_time > 0 else 0.0
        norm = np.linalg.norm(delta_state)
        direction = delta_state / norm if norm > 0 else np.zeros_like(delta_state)

        # Compute rhythm signature using previous movement if available
        if prev_movement is not None and prev_movement.delta_time > 1e-6:
            tempo_ratio = delta_time / prev_movement.delta_time
            velocity_delta = velocity - prev_movement.velocity
            acceleration = velocity_delta / delta_time if delta_time > 1e-6 else 0.0

            if np.linalg.norm(prev_movement.direction) > 1e-6:
                dot = np.clip(np.dot(direction, prev_movement.direction), -1.0, 1.0)
                phase_angle = float(np.arccos(dot))
            else:
                phase_angle = 0.0
        else:
            tempo_ratio = 1.0
            velocity_delta = 0.0
            acceleration = 0.0
            phase_angle = 0.0

        # Entropy approximation based on direction change
        entropy_rate = abs(phase_angle) / np.pi

        rhythm = RhythmSignature(
            tempo_ratio=tempo_ratio,
            velocity_delta=velocity_delta,
            acceleration=acceleration,
            phase_angle=phase_angle,
            entropy_rate=entropy_rate,
        )

        return cls(
            delta_state=delta_state,
            delta_time=delta_time,
            velocity=velocity,
            direction=direction,
            rhythm_signature=rhythm,
            mahalanobis_distance=mahalanobis_dist,
            timestamp=timestamp,
        )