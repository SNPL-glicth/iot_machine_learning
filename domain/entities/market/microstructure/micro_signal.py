"""Predictor short-horizon sobre L1 (FASE 3).

Qué comportamiento de las órdenes precede a los movimientos: el score
combina OFI + agresión en tape + imbalance top-of-book
(pesos documentados 0.5/0.3/0.2). Determinista y sin estado.

La magnitud se ancla a la volatilidad del tramo escalada al horizonte
(σ_h = max(vol trade escalada, piso de medio spread)): el movimiento
esperado no puede exceder en orden lo que el tramo ya muestra.
Los cuantiles son gaussianos sobre (expected, σ_h) — mandan para
pinball/CRPS (Fase 1); ``to_prediction`` deja la Prediction lista
para el ciclo Outcome→Evaluation→Reward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from ..observations import Quote
from ..prediction import Prediction, PredictionInterval, ReturnDistribution
from ..prediction.types import InputContext
from .l1_features import L1Features

__all__ = [
    "MICRO_HORIZONS",
    "MICRO_STRATEGY",
    "MicroSignal",
    "predict_micro",
    "to_prediction",
]

#: Horizontes short-horizon del predictor micro (30s/60s).
MICRO_HORIZONS: Final = (30, 60)

#: Nombre de estrategia para trazabilidad (ContextKey, selección).
MICRO_STRATEGY: Final = "micro-l1"

_W_OFI: float = 0.5
_W_SIGNED: float = 0.3
_W_IMB: float = 0.2
_LOGISTIC_K: float = 3.0
_Z_80: float = 1.2816


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass(frozen=True, slots=True, kw_only=True)
class MicroSignal:
    """Señal microestructural con su distribución (inmutable)."""

    probability_up: float
    expected_return: float
    volatility: float
    quantile_10: float
    quantile_50: float
    quantile_90: float
    score: float  # combinado OFI/agresión/imbalance en [-1, 1] aprox

    def __post_init__(self) -> None:
        for name in ("probability_up",):
            value = getattr(self, name)
            if not 0.05 <= value <= 0.95:
                raise ValueError(f"{name} fuera de [0.05, 0.95]: {value!r}")
        for name in ("expected_return", "volatility", "quantile_10",
                     "quantile_50", "quantile_90", "score"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} no finito")
        if self.volatility <= 0:
            raise ValueError("volatility debe ser > 0")
        if not (self.quantile_10 <= self.quantile_50 <= self.quantile_90):
            raise ValueError("cuantiles desordenados")


def predict_micro(
    features: L1Features, *, horizon_seconds: int = 60
) -> MicroSignal:
    """Señal short-horizon desde el vector L1."""
    if not isinstance(features, L1Features):
        raise TypeError("features debe ser L1Features")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds debe ser > 0")

    score = (
        _W_OFI * features.ofi
        + _W_SIGNED * features.signed_volume_ratio
        + _W_IMB * features.imbalance
    )
    score = max(-1.0, min(1.0, score))
    prob_up = _logistic(_LOGISTIC_K * score)
    prob_up = min(0.95, max(0.05, prob_up))

    span = max(features.span_seconds, 1e-9)
    vol_trades = features.trade_price_vol * math.sqrt(horizon_seconds / span)
    half_spread = abs(features.spread_bps) / 10000.0 / 2.0
    sigma_h = max(vol_trades, half_spread, 1e-9)

    expected = score * sigma_h
    return MicroSignal(
        probability_up=prob_up,
        expected_return=expected,
        volatility=sigma_h,
        quantile_10=expected - _Z_80 * sigma_h,
        quantile_50=expected,
        quantile_90=expected + _Z_80 * sigma_h,
        score=score,
    )


def to_prediction(
    signal: MicroSignal,
    *,
    anchor: Quote,
    prediction_id: str,
    horizon_seconds: int,
    timestamp: float,
    strategy: str = MICRO_STRATEGY,
    feature_version: str = "micro-l1-v1",
) -> Prediction:
    """Construye la Prediction lista para el ciclo (con distribución)."""
    if not isinstance(signal, MicroSignal):
        raise TypeError("signal debe ser MicroSignal")
    if not isinstance(anchor, Quote):
        raise TypeError("anchor debe ser Quote (top-of-book)")
    confidence = max(0.05, min(0.95, 0.5 + abs(signal.score) * 0.4))
    return Prediction(
        prediction_id=prediction_id,
        observation=anchor,
        horizon_seconds=horizon_seconds,
        timestamp=timestamp,
        entry_price=anchor.midpoint,
        expected_return=signal.expected_return,
        probability_up=signal.probability_up,
        confidence=confidence,
        interval=PredictionInterval(
            lower=signal.quantile_10,
            upper=signal.quantile_90,
            confidence_level=0.80,
        ),
        distribution=ReturnDistribution(
            expected_return=signal.expected_return,
            volatility=signal.volatility,
            quantile_10=signal.quantile_10,
            quantile_50=signal.quantile_50,
            quantile_90=signal.quantile_90,
            expected_holding_seconds=horizon_seconds,
        ),
        strategy=strategy,
        input_context=InputContext(
            data_status=anchor.data_status,
            feature_count=13,
            feature_version=feature_version,
        ),
    )
