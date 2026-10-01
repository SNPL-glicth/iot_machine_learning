"""Sentinelas estadísticos genéricos de detección ultrarrápida O(1).

- GenericShockSentinel: Evalúa transiciones y choques instantáneos |dx/dt|.
- GenericRegimeShiftSentinel: Evalúa deriva persistente del centroide o nivel.
"""

from __future__ import annotations

import numpy as np

from .calibrators import EmpiricalDistributionProfile


class GenericShockSentinel:
    """Evalúa sorpresas en la tasa de cambio instantánea |dx/dt|."""

    def __init__(self, shock_profile: EmpiricalDistributionProfile) -> None:
        self.profile = shock_profile
        self.threshold = shock_profile.q_shock_high

    def inspect(self, block_values: np.ndarray, prev_value: float) -> tuple[bool, float]:
        """Evalúa si la velocidad local supera el umbral del cuantil nominal."""
        if len(block_values) == 0:
            return False, 0.0

        v0_diff = abs(block_values[0] - prev_value)
        internal_diffs = (
            np.max(np.abs(np.diff(block_values))) if len(block_values) > 1 else 0.0
        )
        max_dx = max(v0_diff, float(internal_diffs))

        # Sorpresa relativa normalizada por el cuantil nominal
        ratio = max_dx / max(self.threshold, 1e-9)
        surprise_score = float(max(0.0, np.log1p(ratio)))
        triggered = max_dx > self.threshold

        return triggered, round(surprise_score, 4)


class GenericRegimeShiftSentinel:
    """Evalúa sorpresas sostenidas en el centroide o nivel respecto al régimen nominal.

    Distingue conceptualmente:
    - Point surprise: bloque individual fuera de la envolvente extrema [Q_0.01, Q_0.99]
    - Persistent distributional change: k >= 2 bloques consecutivos fuera de [Q_0.10, Q_0.90]
    """

    def __init__(self, level_profile: EmpiricalDistributionProfile) -> None:
        self.profile = level_profile
        self.q_low = level_profile.q_low
        self.q_high = level_profile.q_high
        self.q_warn_low = level_profile.quantiles_raw.get(0.10, level_profile.q_low)
        self.q_warn_high = level_profile.quantiles_raw.get(0.90, level_profile.q_high)
        self.consecutive_warn: int = 0

    def inspect(self, block_values: np.ndarray) -> tuple[bool, float]:
        """Evalúa si la media del bloque manifiesta un cambio persistente de distribución."""
        if len(block_values) == 0:
            return False, 0.0

        block_mean = float(np.mean(block_values))

        # 1. Point surprise extrema
        is_point_out = (block_mean < self.q_low) or (block_mean > self.q_high)

        # 2. Persistent distributional change (k >= 2 bloques fuera del 80% central)
        is_warn_out = (block_mean < self.q_warn_low) or (block_mean > self.q_warn_high)
        if is_warn_out:
            self.consecutive_warn += 1
        else:
            self.consecutive_warn = 0

        is_persistent = self.consecutive_warn >= 2

        triggered = is_point_out or is_persistent

        dist = 0.0
        if block_mean < self.q_warn_low:
            dist = self.q_warn_low - block_mean
        elif block_mean > self.q_warn_high:
            dist = block_mean - self.q_warn_high

        ratio = dist / max(self.profile.interquartile_range, 1e-9)
        surprise_score = float(max(0.0, np.log1p(ratio)))

        return triggered, round(surprise_score, 4)
