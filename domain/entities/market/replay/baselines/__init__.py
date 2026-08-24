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

from .prediction_signal import PredictionSignal
from .predictor import Predictor
from .naive import NaivePredictor
from .momentum import MomentumPredictor
from .ema_crossover import EmaCrossoverPredictor
from .mean_reversion import MeanReversionPredictor
from .random import RandomPredictor

BASELINES: list[Predictor] = [
    NaivePredictor(),
    MomentumPredictor(),
    MomentumPredictor(scale=4.0, lookback_override=10),
    EmaCrossoverPredictor(),
    MeanReversionPredictor(),
    RandomPredictor(),
]

__all__ = [
    "BASELINES",
    "EmaCrossoverPredictor",
    "MeanReversionPredictor",
    "MomentumPredictor",
    "NaivePredictor",
    "PredictionSignal",
    "Predictor",
    "RandomPredictor",
]