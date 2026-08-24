"""Market Replay Engine Classification (regime, signal)."""

from __future__ import annotations

from ..prediction.types import InputContext, PredictionInterval, Regime
from ..data_status import DataStatus
from .baselines import PredictionSignal
from .regime import MarketRegime, classify_window
from .feature_window import FeatureWindow
from .config import ReplayEngineConfig


def classify_regime(window: FeatureWindow) -> Regime | None:
    """Régimen de la ventana (FASE 8): contexto para la adaptación.

    El dominio de predicción usa ``Regime`` (bull/bear/neutral/
    high_volatility); el benchmark clasifica con ``MarketRegime``
    (más fino). Se mapea conservadoramente: la pérdida de detalle
    es intencional y documentada.
    """
    try:
        market = classify_window(window)
    except ValueError:
        return None
    mapping = {
        MarketRegime.TRENDING: Regime.BULL,
        MarketRegime.RANGE: Regime.NEUTRAL,
        MarketRegime.LOW_VOLATILITY: Regime.NEUTRAL,
        MarketRegime.HIGH_VOLATILITY: Regime.HIGH_VOLATILITY,
        MarketRegime.CRASH: Regime.BEAR,
    }
    return mapping[market]


def signal_for(horizon: int, window: FeatureWindow, cfg: ReplayEngineConfig) -> PredictionSignal:
    """Genera señal de predicción para un horizonte."""
    if cfg.predictor is not None:
        return cfg.predictor.predict(
            window,
            horizon_seconds=horizon,
            observation_interval=cfg.interval_seconds,
            lookback=cfg.predictor_lookback,
        )
    from .predictor import predict_direction
    prob, expected, lower, upper, level = predict_direction(
        window,
        horizon_seconds=horizon,
        observation_interval=cfg.interval_seconds,
        lookback=cfg.predictor_lookback,
    )
    return PredictionSignal(
        probability_up=prob,
        expected_return=expected,
        lower=lower,
        upper=upper,
        confidence_level=level,
    )