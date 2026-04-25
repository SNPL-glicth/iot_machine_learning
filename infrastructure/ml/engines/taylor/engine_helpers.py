"""Taylor engine private helpers — extracted to keep engine.py under 180 lines.

All functions are pure or near-pure; any engine-mutation is explicit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult

from .types import DerivativeMethod
from .diagnostics import compute_diagnostic
from .derivatives import estimate_derivatives

if TYPE_CHECKING:
    from .engine import TaylorPredictionEngine

logger = logging.getLogger(__name__)

_VARIANCE_EPSILON: float = 1e-9


def sanitize_inputs(
    values: List[float],
    timestamps: Optional[List[float]],
) -> tuple[List[float], Optional[List[float]]]:
    """Filtra NaN e inf del input. No lanza excepciones."""
    ts = timestamps if timestamps is not None else list(range(len(values)))
    clean = [
        (v, t)
        for v, t in zip(values, ts)
        if v is not None
        and not (v != v)  # NaN check
        and abs(v) != float("inf")
    ]
    if not clean:
        return [], None
    c_values, c_timestamps = zip(*clean)
    return list(c_values), list(c_timestamps) if timestamps is not None else None


def load_hyperparams(engine: "TaylorPredictionEngine", window_size: int) -> None:
    """Load adaptive hyperparameters from the HyperparameterAdaptor."""
    if engine._hyperparams is None or not engine._series_id:
        return
    params = engine._hyperparams.load(engine._series_id, engine.name)
    if not params:
        return
    order_raw = params.get("order")
    if order_raw is not None:
        try:
            new_order = int(order_raw)
        except (TypeError, ValueError):
            new_order = engine._order
        engine._order = max(1, min(new_order, 3))
    logger.debug(
        "hyperparams_loaded series=%s engine=%s order=%d window=%d",
        engine._series_id,
        engine.name,
        engine._order,
        window_size,
    )


def compute_taylor_coefficients(
    values: List[float],
    dt: float,
    base_order: int,
    method: DerivativeMethod,
) -> object:
    """Compute Taylor coefficients with variance check (CRIT-4)."""
    variance = compute_variance(values)
    effective_order = base_order
    if variance < _VARIANCE_EPSILON:
        effective_order = 0
        logger.debug(
            "taylor_order_reduced_to_zero",
            extra={
                "variance": variance,
                "threshold": _VARIANCE_EPSILON,
                "reason": "near_constant_signal",
            },
        )
    return estimate_derivatives(values, dt, effective_order, method)


def classify_trend(slope: float, threshold: float) -> str:
    if slope > threshold:
        return "up"
    if slope < -threshold:
        return "down"
    return "stable"


def confidence_from_stability(
    stability: float, min_confidence: float, max_confidence: float
) -> float:
    instability = min(stability, 0.7)
    c = max(min_confidence, 1.0 - instability)
    return min(c, max_confidence)


def compute_variance(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def taylor_fallback(
    values: List[float],
    min_confidence: float,
    horizon: int,
) -> PredictionResult:
    tail = values[-min(3, len(values)):]
    predicted = sum(tail) / len(tail)
    logger.debug("taylor_fallback", extra={"n": len(values)})
    return PredictionResult(
        predicted_value=predicted,
        confidence=max(min_confidence, min(0.5, len(values) / 10.0)),
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
            "horizon_steps": horizon,
            "fallback": "insufficient_data",
            "clamped": False,
            "diagnostic": None,
        },
    )
