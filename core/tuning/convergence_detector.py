"""Detección de convergencia y oscilación en parámetros adaptativos.

Principio: Single Responsibility - solo detecta convergencia/oscilación.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np


class ConvergenceStatus(Enum):
    """Estados de convergencia de parámetros adaptativos."""
    CONVERGING = "converging"
    CONVERGED = "converged"
    OSCILLATING = "oscillating"
    DIVERGING = "diverging"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class ConvergenceResult:
    """Resultado del análisis de convergencia."""
    status: ConvergenceStatus
    current_value: float
    delta_mean: float
    delta_std: float
    oscillation_count: int
    steps_since_change: int
    recommendation: str


class ConvergenceDetector:
    """
    Detecta convergencia/oscilación en series de valores adaptativos.

    Uso:
        detector = ConvergenceDetector(window=20, threshold=1e-4)
        result = detector.update(new_alpha_value)
        if result.status == ConvergenceStatus.OSCILLATING:
            # reducir learning rate
    """

    def __init__(
        self,
        window: int = 20,
        convergence_threshold: float = 1e-4,
        oscillation_threshold: int = 3,
        divergence_factor: float = 2.0,
    ) -> None:
        self._window = window
        self._convergence_threshold = convergence_threshold
        self._oscillation_threshold = oscillation_threshold
        self._divergence_factor = divergence_factor

        self._values: deque = deque(maxlen=window)
        self._deltas: deque = deque(maxlen=window - 1)
        self._steps_since_significant_change = 0

    def update(self, value: float) -> ConvergenceResult:
        """Agrega nuevo valor y retorna status de convergencia."""
        self._values.append(value)

        if len(self._values) < 2:
            return ConvergenceResult(
                status=ConvergenceStatus.INSUFFICIENT_DATA,
                current_value=value,
                delta_mean=0.0,
                delta_std=0.0,
                oscillation_count=0,
                steps_since_change=0,
                recommendation="need_more_data",
            )

        # Compute delta
        delta = self._values[-1] - self._values[-2]
        self._deltas.append(delta)

        # Check if change is significant
        if abs(delta) < self._convergence_threshold:
            self._steps_since_significant_change += 1
        else:
            self._steps_since_significant_change = 0

        # Compute statistics
        deltas_array = np.array(self._deltas)
        delta_mean = np.mean(np.abs(deltas_array))
        delta_std = np.std(deltas_array)

        # Count oscillations (sign changes)
        oscillation_count = 0
        if len(self._deltas) >= 2:
            signs = np.sign(deltas_array)
            oscillation_count = int(np.sum(np.diff(signs) != 0))

        # Determine status
        status = self._classify_status(
            delta_mean, delta_std, oscillation_count, len(self._deltas)
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(status, oscillation_count)

        return ConvergenceResult(
            status=status,
            current_value=value,
            delta_mean=delta_mean,
            delta_std=delta_std,
            oscillation_count=oscillation_count,
            steps_since_change=self._steps_since_significant_change,
            recommendation=recommendation,
        )

    def _classify_status(
        self, delta_mean: float, delta_std: float, oscillation_count: int, n_deltas: int
    ) -> ConvergenceStatus:
        """Clasifica el estado de convergencia."""
        if n_deltas < 5:
            return ConvergenceStatus.INSUFFICIENT_DATA

        # Converged: small changes for multiple steps
        if (
            delta_mean < self._convergence_threshold
            and self._steps_since_significant_change >= 5
        ):
            return ConvergenceStatus.CONVERGED

        # Oscillating: frequent sign changes
        if oscillation_count >= self._oscillation_threshold:
            return ConvergenceStatus.OSCILLATING

        # Diverging: increasing deltas
        if n_deltas >= 10:
            recent_deltas = list(self._deltas)[-10:]
            early_mean = np.mean(np.abs(recent_deltas[:5]))
            late_mean = np.mean(np.abs(recent_deltas[5:]))
            if late_mean > early_mean * self._divergence_factor:
                return ConvergenceStatus.DIVERGING

        # Default: converging
        return ConvergenceStatus.CONVERGING

    def _generate_recommendation(
        self, status: ConvergenceStatus, oscillation_count: int
    ) -> str:
        """Genera recomendación basada en status."""
        if status == ConvergenceStatus.CONVERGED:
            return "parameter_stable_no_action"
        elif status == ConvergenceStatus.OSCILLATING:
            return f"reduce_learning_rate_oscillations={oscillation_count}"
        elif status == ConvergenceStatus.DIVERGING:
            return "reduce_learning_rate_diverging"
        elif status == ConvergenceStatus.CONVERGING:
            return "continue_monitoring"
        else:
            return "insufficient_data"

    def reset(self) -> None:
        """Reset del detector."""
        self._values.clear()
        self._deltas.clear()
        self._steps_since_significant_change = 0

    def get_history(self) -> List[float]:
        """Retorna historial de valores."""
        return list(self._values)
