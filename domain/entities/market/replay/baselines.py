"""Baselines del benchmark de ZENIN (FASE 5.5).

Cada baseline implementa ``Predictor`` y produce una ``PredictionSignal``
con el mismo contrato del predictor de referencia: el Market Replay los
trata idénticos, de modo que la comparación es justa (misma ventana,
mismo reloj, misma resolución de outcomes).

Los baselines existen para ser superados: si ZENIN no gana de forma
consistente a Naive/Random/Momentum/EMA/Mean-reversion, el marcador
debe decirlo.

Determinismo: todos los baselines son puros; ``RandomPredictor`` usa un
RNG sembrado (reproducible run a run).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from .feature_window import FeatureWindow


@dataclass(frozen=True, slots=True)
class PredictionSignal:
    """Señal mínima que un predictor entrega al engine."""

    probability_up: float
    expected_return: float
    lower: float
    upper: float
    confidence_level: float = 0.50

    def __post_init__(self) -> None:
        if not 0.05 <= self.probability_up <= 0.95:
            raise ValueError(
                f"probability_up fuera de [0.05, 0.95]: {self.probability_up!r}"
            )
        if not math.isfinite(self.expected_return):
            raise ValueError("expected_return debe ser finito")
        if not self.lower <= self.expected_return <= self.upper:
            raise ValueError(
                "expected_return debe estar contenido en [lower, upper]"
            )


class Predictor(Protocol):
    """Contrato de un predictor evaluable por el benchmark."""

    name: str

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        ...


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _clamp_prob(value: float) -> float:
    return min(0.95, max(0.05, value))


def _band(expected: float, vol: float, horizon_ratio: float) -> tuple[float, float]:
    half = 0.6745 * vol * math.sqrt(horizon_ratio)
    return expected - half, expected + half


class NaivePredictor:
    """Mañana = último precio (martingala). Sin señal direccional."""

    name = "naive"

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        return PredictionSignal(
            probability_up=0.50,
            expected_return=0.0,
            lower=-1e-9,
            upper=1e-9,
            confidence_level=0.50,
        )


class MomentumPredictor:
    """Dirección del drift de los últimos ``lookback`` retornos."""

    def __init__(self, *, scale: float = 2.0, lookback_override: int | None = None) -> None:
        self._scale = scale
        self._lookback_override = lookback_override
        self.name = "momentum"

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        lb = self._lookback_override or lookback
        if window.size < lb + 1:
            raise ValueError(f"ventana insuficiente para momentum ({lb + 1} velas)")
        drift = window.mean_return(lb)
        vol = window.std_return(lb) or 1e-12
        ratio = horizon_seconds / observation_interval
        prob_up = _clamp_prob(_logistic(drift / vol * math.sqrt(lb) * self._scale))
        expected = drift * ratio
        lower, upper = _band(expected, vol, ratio)
        return PredictionSignal(
            probability_up=prob_up, expected_return=expected, lower=lower, upper=upper
        )


class EmaCrossoverPredictor:
    """Cruce de EMAs (rápida vs lenta) sobre closes de la ventana."""

    def __init__(self, *, fast: int = 4, slow: int = 12) -> None:
        if not 1 <= fast < slow:
            raise ValueError("se requiere 1 <= fast < slow")
        self._fast = fast
        self._slow = slow
        self.name = "ema-crossover"

    @staticmethod
    def _ema(values: tuple[float, ...], span: int) -> float:
        alpha = 2.0 / (span + 1.0)
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        return ema

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        closes = tuple(c.close for c in window.candles)
        if len(closes) < self._slow + 1:
            raise ValueError(
                f"ventana insuficiente para ema-crossover ({self._slow} velas)"
            )
        fast_ema = self._ema(closes, self._fast)
        slow_ema = self._ema(closes, self._slow)
        if slow_ema <= 0:
            raise ValueError("closes deben ser > 0 para ema-crossover")
        spread = (fast_ema - slow_ema) / slow_ema
        ratio = horizon_seconds / observation_interval
        prob_up = _clamp_prob(_logistic(spread * 40.0))
        expected = spread * ratio * 0.5
        vol = window.std_return(self._slow) or 1e-12
        lower, upper = _band(expected, vol, ratio)
        return PredictionSignal(
            probability_up=prob_up, expected_return=expected, lower=lower, upper=upper
        )


class MeanReversionPredictor:
    """Reversión a la media: el precio lejos de su EMA lenta tiende a volver."""

    def __init__(self, *, span: int = 12) -> None:
        self._span = span
        self.name = "mean-reversion"

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        closes = tuple(c.close for c in window.candles)
        if len(closes) < self._span + 1:
            raise ValueError(
                f"ventana insuficiente para mean-reversion ({self._span} velas)"
            )
        alpha = 2.0 / (self._span + 1.0)
        ema = closes[0]
        for value in closes[1:]:
            ema = alpha * value + (1 - alpha) * ema
        if ema <= 0:
            raise ValueError("ema debe ser > 0 para mean-reversion")
        deviation = (closes[-1] - ema) / ema
        ratio = horizon_seconds / observation_interval
        prob_up = _clamp_prob(_logistic(-deviation * 40.0))
        expected = -deviation * ratio * 0.5
        vol = window.std_return(self._span) or 1e-12
        lower, upper = _band(expected, vol, ratio)
        return PredictionSignal(
            probability_up=prob_up, expected_return=expected, lower=lower, upper=upper
        )


class RandomPredictor:
    """Moneda con ruido sembrado: cota honesta de "señal nula"."""

    def __init__(self, *, seed: int = 20260816, noise: float = 0.05) -> None:
        self._rng = random.Random(seed)
        self._noise = noise
        self.name = "random"

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        prob_up = _clamp_prob(0.50 + self._rng.uniform(-self._noise, self._noise))
        return PredictionSignal(
            probability_up=prob_up,
            expected_return=0.0,
            lower=-1e-9,
            upper=1e-9,
            confidence_level=0.50,
        )


BASELINES: list[Predictor] = [
    NaivePredictor(),
    MomentumPredictor(),
    MomentumPredictor(scale=4.0, lookback_override=10),
    EmaCrossoverPredictor(),
    MeanReversionPredictor(),
    RandomPredictor(),
]
