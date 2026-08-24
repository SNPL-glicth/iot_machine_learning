"""Clasificación de régimen de mercado (FASE 5.5).

Determinista y puro: clasifica una ventana de velas cerradas en uno de
los cinco regímenes del benchmark. Umbrales fijos y documentados; el
régimen de un período se asigna con las primeras velas del período
(walk-forward: el régimen del test se conoce revisando lo ya ocurrido).
"""

from __future__ import annotations

import math
from enum import Enum

from .feature_window import FeatureWindow


class MarketRegime(Enum):
    TRENDING = "TRENDING"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    CRASH = "CRASH"
    LOW_VOLATILITY = "LOW_VOLATILITY"


def classify_window(
    window: FeatureWindow,
    *,
    lookback: int = 20,
    ema_span: int = 12,
    crash_z: float = 0.75,
    trend_z: float = 0.50,
    high_vol_rel: float = 0.010,
    low_vol_rel: float = 0.004,
) -> MarketRegime:
    """Clasifica la ventana; requiere lookback + 1 velas cerradas."""
    if window.size < lookback + 1:
        raise ValueError(
            f"ventana insuficiente para régimen: {window.size} < {lookback + 1}"
        )
    returns = window.returns(lookback)
    drift = sum(returns) / len(returns)
    var = sum((r - drift) ** 2 for r in returns) / len(returns)
    vol = math.sqrt(var)
    if vol <= 0.0:
        vol = 1e-12

    closes = tuple(c.close for c in window.candles)
    last = closes[-1]
    mean_close = sum(closes[-lookback:]) / lookback
    vol_rel = vol / mean_close

    alpha = 2.0 / (ema_span + 1.0)
    ema = closes[0]
    for value in closes[1:]:
        ema = alpha * value + (1 - alpha) * ema

    short_drift = sum(returns[-5:]) / min(5, len(returns))
    signal = short_drift / vol * math.sqrt(len(returns))
    below_ema = last < ema

    # Prioridades: CRASH > HIGH_VOL > LOW_VOL > TRENDING > RANGE
    if below_ema and signal <= -crash_z:
        return MarketRegime.CRASH
    if vol_rel >= high_vol_rel:
        return MarketRegime.HIGH_VOLATILITY
    if vol_rel <= low_vol_rel:
        return MarketRegime.LOW_VOLATILITY
    if abs(signal) >= trend_z:
        return MarketRegime.TRENDING
    return MarketRegime.RANGE
