"""Predictor de referencia del Market Replay (FASE 5).

Un predictor **determinista** (sin RNG, sin estado oculto): las mismas
velas cerradas producen exactamente la misma predicción. Eso hace que el
test de oro anti-look-ahead sea binario: si el feed se corta después de
la predicción, la predicción no puede cambiar.

Modelo de referencia (honesto, no "genio"):
    * drift  = media de retornos de la ventana corta
    * vol    = desviación estándar de retornos
    * prob_up = sigmoide(drift * sqrt(n) / vol), recortada a [0.05, 0.95]
    * expected_return = drift escalado linealmente al horizonte
    * intervalo con nivel de confianza fijo 0.50 (z = 0.6745)

Es un baseline: las features reales y los modelos (MoE) vienen después;
este predictor existe para que el pipeline completo sea verificable hoy.
"""

from __future__ import annotations

import math

from .feature_window import FeatureWindow

_SIGMOID_SCALE = 2.0
_Z_50 = 0.6745


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def predict_direction(
    window: FeatureWindow,
    *,
    horizon_seconds: int,
    observation_interval: int,
    lookback: int = 20,
) -> tuple[float, float, float, float, float]:
    """Predice (prob_up, expected_return, lower, upper, confidence_level).

    Devuelve el pronóstico completo con que se construye la Prediction.
    Requiere ventana con al menos ``lookback + 1`` velas cerradas.
    """
    if window.size < lookback + 1:
        raise ValueError(
            f"ventana insuficiente: {window.size} velas, "
            f"se requieren {lookback + 1}"
        )
    drift = window.mean_return(lookback)
    vol = window.std_return(lookback)
    if vol == 0.0:
        vol = 1e-12
    scale = math.sqrt(lookback) * _SIGMOID_SCALE
    prob_up = _logistic(drift / vol * scale)
    prob_up = min(0.95, max(0.05, prob_up))
    expected = drift * (horizon_seconds / observation_interval)
    half_width = _Z_50 * vol * math.sqrt(horizon_seconds / observation_interval)
    lower = expected - half_width
    upper = expected + half_width
    return prob_up, expected, lower, upper, 0.50
