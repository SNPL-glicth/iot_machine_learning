"""Prediction Guards (FASE 3)."""

from __future__ import annotations

from .outcome import Outcome
from .prediction_entity import Prediction


def guard_outcome(prediction: Prediction, outcome: Outcome) -> None:
    """Valida que el Outcome corresponde a esta Prediction."""
    from .outcome import Outcome as _Outcome

    if not isinstance(outcome, _Outcome):
        raise TypeError("outcome debe ser Outcome")
    if outcome.symbol != prediction.observation.symbol:
        raise ValueError(
            "outcome de otro símbolo: "
            f"{outcome.symbol!r} vs {prediction.observation.symbol!r}"
        )
    if outcome.horizon_seconds != prediction.horizon_seconds:
        raise ValueError(
            "outcome de otro horizonte: "
            f"{outcome.horizon_seconds}s vs predicción {prediction.horizon_seconds}s"
        )