"""Market Replay Engine Result."""

from __future__ import annotations

from dataclasses import dataclass

from ..prediction.prediction import Prediction
from ..prediction.outcome import Outcome

__all__ = ["ReplayRunResult"]


@dataclass(frozen=True, slots=True)
class ReplayRunResult:
    """Resultado completo e inmutable de un run del replay."""

    symbol: str
    predictions: tuple[Prediction, ...] = ()
    outcomes: tuple[Outcome, ...] = ()
    invalidated: tuple[Prediction, ...] = ()
    latency_ns: tuple[int, ...] = ()