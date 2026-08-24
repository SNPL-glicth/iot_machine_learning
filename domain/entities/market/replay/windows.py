"""Walk-forward window types and generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..observations import Candle
from .feature_window import FeatureWindow
from .regime import MarketRegime, classify_window

__all__ = ["WfWindow", "wf_windows", "window_regime"]

_REGIME_LABELS: dict[MarketRegime, str] = {
    MarketRegime.TRENDING: "bull",
    MarketRegime.RANGE: "neutral",
    MarketRegime.LOW_VOLATILITY: "neutral",
    MarketRegime.HIGH_VOLATILITY: "high_volatility",
    MarketRegime.CRASH: "bear",
}

_REGIME_LOOKBACK = 21


@dataclass(frozen=True, slots=True)
class WfWindow:
    """Un par (TRAIN, TEST) contiguo y disjunto con sus spans."""

    index: int
    train: tuple[Candle, ...]
    test: tuple[Candle, ...]
    train_start: float
    train_end: float
    test_start: float
    test_end: float


def wf_windows(
    candles: Sequence[Candle],
    *,
    train_seconds: float,
    test_seconds: float,
    step_seconds: float,
    min_train: int = 60,
    min_test: int = 2,
) -> tuple[WfWindow, ...]:
    """Ventanas walk-forward (origin rolling) con mínimo honrado.

    Nada del TEST alimenta el TRAIN de ninguna ventana: los spans son
    disjuntos y el test siempre sigue al train en el tiempo.
    """
    candles = tuple(candles)
    if not candles:
        return ()
    start = candles[0].timestamp
    end = candles[-1].timestamp
    windows: list[WfWindow] = []
    origin = start
    index = 0
    while True:
        train_end = origin + train_seconds
        test_end = train_end + test_seconds
        train = tuple(c for c in candles if origin <= c.timestamp < train_end)
        test = tuple(c for c in candles if train_end <= c.timestamp < test_end)
        if len(train) < min_train or len(test) < min_test:
            break
        windows.append(
            WfWindow(
                index=index,
                train=train,
                test=test,
                train_start=origin,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
            )
        )
        index += 1
        origin += step_seconds
        if origin >= end:
            break
    return tuple(windows)


def window_regime(candles: Sequence[Candle]) -> str | None:
    """Régimen que el modelo ve al entrar al TEST (cola del TRAIN).

    Coincide con la etiqueta que el engine escribe en las predicciones
    (bull/bear/neutral/high_volatility), así el contexto de la versión
    del modelo cuadra con el historial del store.
    """
    candles = tuple(candles)
    if len(candles) < _REGIME_LOOKBACK:
        return None
    try:
        window = FeatureWindow(
            symbol=candles[-1].symbol,
            candles=candles[-_REGIME_LOOKBACK:],
        )
        regime = classify_window(window)
    except ValueError:
        return None
    return _REGIME_LABELS.get(regime)