"""Guardrail de seguridad para la calibración adaptativa.

Evita la normalización de fallas y deterioros graduales controlando el estado
de congelamiento de la calibración.
"""
from __future__ import annotations

from collections import deque
from enum import Enum


class CalibrationState(str, Enum):
    """Estados del ciclo de calibración adaptativa."""

    NOMINAL_STABLE = "NOMINAL_STABLE"
    DRIFT_FROZEN = "DRIFT_FROZEN"
    ANOMALY_BLOCKED = "ANOMALY_BLOCKED"


class CalibrationGuardrail:
    """Controlador causal de estados de calibración."""

    def __init__(
        self,
        drift_window: int = 20,
        persistence_threshold: int = 12,
        min_gradient: float = 1e-5,
    ) -> None:
        self._drift_window = drift_window
        self._persistence_threshold = persistence_threshold
        self._min_gradient = min_gradient
        self._recent_diffs: deque[float] = deque(maxlen=drift_window)
        self._state: CalibrationState = CalibrationState.NOMINAL_STABLE
        self._last_value: float | None = None

    @property
    def state(self) -> CalibrationState:
        return self._state

    def is_adaptation_allowed(self) -> bool:
        """Determina si se permite actualizar o recalibrar umbrales."""
        return self._state == CalibrationState.NOMINAL_STABLE

    def observe(self, value: float, is_anomaly_active: bool = False) -> CalibrationState:
        """Actualiza el estado causalmente a partir de la nueva observación.

        Args:
            value: Valor observado en el instante t.
            is_anomaly_active: True si el ensamble reporta anomalía en t.

        Returns:
            Estado actual de calibración.
        """
        if is_anomaly_active:
            self._state = CalibrationState.ANOMALY_BLOCKED
            return self._state

        if self._last_value is not None:
            self._recent_diffs.append(value - self._last_value)
        else:
            self._recent_diffs.append(0.0)

        self._last_value = value

        # Evaluar persistencia monotónica unidireccional (deriva térmica/desgaste)
        if len(self._recent_diffs) >= self._drift_window:
            positives = sum(1 for d in self._recent_diffs if d > self._min_gradient)
            negatives = sum(1 for d in self._recent_diffs if d < -self._min_gradient)

            if positives >= self._persistence_threshold or negatives >= self._persistence_threshold:
                self._state = CalibrationState.DRIFT_FROZEN
                return self._state

        self._state = CalibrationState.NOMINAL_STABLE
        return self._state

    def reset(self) -> None:
        """Reinicia el historial de derivas del guardrail."""
        self._recent_diffs.clear()
        if hasattr(self, "_last_value"):
            del self._last_value
        self._state = CalibrationState.NOMINAL_STABLE
