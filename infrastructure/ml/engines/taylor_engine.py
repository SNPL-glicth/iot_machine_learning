"""Motor de predicción Taylor — orquesta cálculos de taylor_math.py.

Migrado desde ml/core/taylor_predictor.py.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from iot_machine_learning.domain.validators.numeric import (
    ValidationError,
    clamp_prediction,
    validate_window,
)
from iot_machine_learning.infrastructure.ml.interfaces import PredictionEngine, PredictionResult

from .taylor_math import (
    compute_accel_variance,
    compute_dt,
    compute_finite_differences,
    taylor_expand,
)

logger = logging.getLogger(__name__)

# Constantes de configuración
_TREND_THRESHOLD: float = 0.01
_MIN_CONFIDENCE: float = 0.3
_MAX_CONFIDENCE: float = 0.95
_CLAMP_MARGIN_PCT: float = 0.3


class TaylorPredictionEngine(PredictionEngine):
    """Motor de predicción basado en Series de Taylor.

    Calcula derivadas numéricas (velocidad, aceleración, jerk) y usa la
    expansión de Taylor para extrapolar el siguiente valor.

    Attributes:
        _order: Orden máximo de Taylor (1–3).
        _horizon: Pasos adelante a predecir (en unidades de Δt).
    """

    def __init__(self, order: int = 2, horizon: int = 1) -> None:
        if horizon < 1:
            raise ValueError(f"horizon debe ser >= 1, recibido {horizon}")

        self._order: int = max(1, min(order, 3))
        self._horizon: int = horizon

        if order != self._order:
            logger.warning(
                "taylor_order_clamped",
                extra={"requested": order, "effective": self._order},
            )

    @property
    def name(self) -> str:
        return "taylor_finite_differences"

    def can_handle(self, n_points: int) -> bool:
        return n_points >= self._order + 2

    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        validate_window(values, min_size=1)

        n = len(values)
        dt = compute_dt(timestamps)

        if not self.can_handle(n):
            return self._fallback_prediction(values)

        effective_order = self._order
        while effective_order > 0 and n < effective_order + 1:
            effective_order -= 1

        if effective_order == 0:
            return self._fallback_prediction(values)

        derivs = compute_finite_differences(values, dt, effective_order)
        h = float(self._horizon) * dt
        predicted_raw = taylor_expand(derivs, h, effective_order)

        predicted, was_clamped = clamp_prediction(
            predicted_raw, values, margin_pct=_CLAMP_MARGIN_PCT
        )

        f_prime = derivs["f_prime"]
        if f_prime > _TREND_THRESHOLD:
            trend = "up"
        elif f_prime < -_TREND_THRESHOLD:
            trend = "down"
        else:
            trend = "stable"

        confidence = self._compute_confidence(values, dt, derivs["f_t"])

        metadata: dict = {
            "order": effective_order,
            "derivatives": {
                "f_t": derivs["f_t"],
                "f_prime": derivs["f_prime"],
                "f_double_prime": derivs["f_double_prime"],
                "f_triple_prime": derivs["f_triple_prime"],
            },
            "dt": dt,
            "horizon_steps": self._horizon,
            "fallback": None,
            "clamped": was_clamped,
        }

        logger.debug(
            "taylor_prediction",
            extra={
                "n_points": n,
                "order": self._order,
                "effective_order": effective_order,
                "f_prime": f_prime,
                "predicted_raw": predicted_raw,
                "predicted_value": predicted,
                "clamped": was_clamped,
                "confidence": confidence,
            },
        )

        return PredictionResult(
            predicted_value=predicted,
            confidence=confidence,
            trend=trend,
            metadata=metadata,
        )

    def supports_uncertainty(self) -> bool:
        return False

    def _fallback_prediction(self, values: List[float]) -> PredictionResult:
        tail = values[-min(3, len(values)):]
        predicted = sum(tail) / len(tail)
        logger.debug("taylor_fallback", extra={"n_points": len(values), "predicted": predicted})
        return PredictionResult(
            predicted_value=predicted,
            confidence=max(_MIN_CONFIDENCE, min(0.5, len(values) / 10.0)),
            trend="stable",
            metadata={
                "order": 0,
                "derivatives": {
                    "f_t": values[-1] if values else 0.0,
                    "f_prime": 0.0,
                    "f_double_prime": 0.0,
                    "f_triple_prime": 0.0,
                },
                "dt": 1.0,
                "horizon_steps": self._horizon,
                "fallback": "insufficient_data",
                "clamped": False,
            },
        )

    def _compute_confidence(
        self,
        values: List[float],
        dt: float,
        f_t: float,
    ) -> float:
        accel_var = compute_accel_variance(values, dt)
        normalizer = abs(f_t) if abs(f_t) > 1e-6 else 1.0
        instability = min(accel_var / normalizer, 0.7)
        confidence = max(_MIN_CONFIDENCE, 1.0 - instability)
        confidence = min(confidence, _MAX_CONFIDENCE)
        return confidence
