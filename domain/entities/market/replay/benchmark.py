"""Walk-forward del benchmark de ZENIN (FASE 5.5).

``split_walk_forward`` divide las velas en ventanas TRAIN -> TEST que
avanzan en el tiempo (origin rolling): ningún TEST alimenta ningún
TRAIN posterior, y el ajuste de parámetros ocurre solo sobre TRAIN.

``TrainedMomentumPredictor`` hace grid-search honesto: evalúa cada
combinación de parámetros exclusivamente sobre el TRAIN (señales
puntuales sin futuro) y el TEST corre con los parámetros ganadores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..observations import Candle
from .baselines import MomentumPredictor, PredictionSignal
from .feature_window import FeatureWindow

_GRID_LOOKBACKS = (5, 10, 20, 40)
_GRID_SCALES = (1.0, 2.0, 4.0)


@dataclass(frozen=True, slots=True)
class TrainTestSplit:
    """Un par (TRAIN, TEST) de velas disjuntas y contiguas en el tiempo."""

    train: tuple[Candle, ...]
    test: tuple[Candle, ...]


def split_walk_forward(
    candles: tuple[Candle, ...],
    *,
    train_seconds: float,
    test_seconds: float,
    step_seconds: float,
    min_train: int = 60,
) -> tuple[TrainTestSplit, ...]:
    """Genera ventanas que avanzan; mínimo honrado para TRAIN."""
    if not candles:
        return ()
    start = candles[0].timestamp
    end = candles[-1].timestamp
    splits: list[TrainTestSplit] = []
    origin = start
    while True:
        train_end = origin + train_seconds
        test_end = train_end + test_seconds
        train = tuple(
            c for c in candles if origin <= c.timestamp < train_end
        )
        test = tuple(
            c for c in candles if train_end <= c.timestamp < test_end
        )
        if len(train) < min_train or not test:
            break
        splits.append(TrainTestSplit(train=train, test=test))
        origin += step_seconds
        if origin >= end:
            break
    return tuple(splits)


class TrainedMomentumPredictor:
    """Momentum con parámetros elegidos por grid-search sobre TRAIN."""

    def __init__(self) -> None:
        self.name = "trained-momentum"
        self._lookback: int = 20
        self._scale: float = 2.0

    def fit(self, train: tuple[Candle, ...], *, horizon_seconds: int) -> None:
        """Elige (lookback, scale) maximizando aciertos de DIRECCIÓN en TRAIN.

        La señal se calcula con velas cerradas <= t (sin futuro); el
        acierto se mide contra el retorno [t, t + horizon] del TRAIN.
        """
        best_score = -math.inf
        best = (self._lookback, self._scale)
        index = {c.timestamp: i for i, c in enumerate(train)}
        horizons_end = {
            c.timestamp: c.timestamp + horizon_seconds for c in train
        }
        for lb in _GRID_LOOKBACKS:
            for scale in _GRID_SCALES:
                score = 0.0
                count = 0
                for i in range(lb, len(train) - 1):
                    candle = train[i]
                    future_ts = horizons_end[candle.timestamp]
                    future = index.get(future_ts)
                    if future is None:
                        continue
                    realized = (train[future].close - candle.close) / candle.close
                    window = FeatureWindow(symbol=train[0].symbol,
                                           candles=train[i - lb + 1: i + 1])
                    if window.size < lb + 1:
                        continue
                    drift = window.mean_return(lb)
                    vol = window.std_return(lb) or 1e-12
                    signal = drift / vol * math.sqrt(lb) * scale
                    predicted_up = signal >= 0
                    count += 1
                    if predicted_up == (realized > 0):
                        score += 1.0
                rate = score / count if count else -math.inf
                if rate > best_score:
                    best_score = rate
                    best = (lb, scale)
        self._lookback, self._scale = best

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        return MomentumPredictor(
            scale=self._scale, lookback_override=self._lookback
        ).predict(
            window,
            horizon_seconds=horizon_seconds,
            observation_interval=observation_interval,
            lookback=lookback,
        )
