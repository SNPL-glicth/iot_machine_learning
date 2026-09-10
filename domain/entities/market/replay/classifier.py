"""Market Replay Engine Classification (regime, signal)."""

from __future__ import annotations

from ..prediction.types import InputContext, PredictionInterval, Regime
from ..data_status import DataStatus
from ..regime import LatentRegime, RegimePosterior, regime_features, score_posterior
from .baselines import PredictionSignal
from .regime import MarketRegime, classify_window
from .feature_window import FeatureWindow
from .config import ReplayEngineConfig

#: Mapeo latente → Regime de predicción (FASE 2). TRENDING/BREAKOUT son
#: agnósticos a dirección y se mapean a BULL como el determinista
#: (pérdida de detalle intencional y documentada).
LATENT_TO_REGIME = {
    LatentRegime.TRENDING: Regime.BULL,
    LatentRegime.BREAKOUT: Regime.BULL,
    LatentRegime.MEAN_REVERTING: Regime.NEUTRAL,
    LatentRegime.LOW_VOL: Regime.NEUTRAL,
    LatentRegime.HIGH_VOL: Regime.HIGH_VOLATILITY,
    LatentRegime.CRASH: Regime.BEAR,
}


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


def latent_posterior(
    window: FeatureWindow, *, lookback: int = 20, temperature: float = 1.0
) -> RegimePosterior | None:
    """Posterior latente de la ventana (FASE 2, blanda, 6 estados).

    Retorna None si la ventana es insuficiente. Solo usa velas
    cerradas ya presentes (walk-forward honesto: el régimen del test
    se conoce revisando lo ya ocurrido).
    """
    try:
        returns = window.returns(lookback)
    except ValueError:
        return None
    return score_posterior(regime_features(returns), temperature=temperature)


def classify_latent_regime(
    window: FeatureWindow, *, lookback: int = 20, temperature: float = 1.0
) -> Regime | None:
    """MAP latente mapeado a ``Regime`` (None sin evidencia)."""
    posterior = latent_posterior(window, lookback=lookback, temperature=temperature)
    if posterior is None:
        return None
    return LATENT_TO_REGIME[posterior.most_likely]


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