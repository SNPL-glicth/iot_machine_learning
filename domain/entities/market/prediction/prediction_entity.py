"""Prediction Entity (FASE 3)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ...market.validators import (
    validate_price,
    validate_timestamp,
    validate_unit_interval,
)
from .lifecycle import (
    InvalidTransitionError,
    PredictionStatus,
    validate_state_consistency,
    validate_transition,
)
from .types import InputContext, PredictionInterval, Regime
from .validation import (
    validate_expected_return,
    validate_horizon,
    validate_interval_contains,
)

if TYPE_CHECKING:
    from ...market import MarketObservation
    from .evaluation import Evaluation
    from .outcome import Outcome
    from .reward import Reward, RewardConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class Prediction:
    """Predicción de mercado sobre una observación (multi-horizonte).

    Una misma observación (ej. 14:30) puede tener varias **Prediction**
    independientes, una por horizonte (1m, 5m, 15m): cada una vive su
    propio ciclo de vida y el cierre de una jamás afecta a las otras.

    Retornos como fracciones (0.05 == 5%). ``entry_price`` es el precio
    de referencia al que se mide el resultado (derivado de la observación
    por el pipeline, ej. midpoint de un Quote).

    El reward solo puede materializarse en la transición
    ``EVALUATED -> REWARDED`` (ver ``lifecycle``).
    """

    prediction_id: str
    observation: "MarketObservation"
    horizon_seconds: int
    timestamp: float
    entry_price: float
    expected_return: float
    probability_up: float
    confidence: float
    interval: PredictionInterval | None = None
    regime: Regime | None = None
    strategy: str | None = None
    input_context: InputContext | None = None

    status: PredictionStatus = PredictionStatus.PENDING
    outcome: "Outcome" | None = None
    evaluation: "Evaluation" | None = None
    reward: "Reward" | None = None
    invalidation_reason: str | None = None  # FASE 6: reason de invalidación (ej. provider_gap)

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        from ...market import MarketObservation

        prediction_id = self.prediction_id.strip()
        if not prediction_id:
            raise ValueError("prediction_id no puede ser vacío")

        if not isinstance(self.observation, MarketObservation):
            raise TypeError("observation debe ser una MarketObservation")

        validate_horizon(self.horizon_seconds)
        validate_timestamp(self.timestamp)
        if self.timestamp < self.observation.timestamp:
            raise ValueError(
                "timestamp de la predicción no puede ser anterior al de su "
                f"observación: {self.timestamp} < {self.observation.timestamp}"
            )

        validate_price(self.entry_price, "entry_price")
        validate_expected_return(self.expected_return)
        validate_unit_interval(self.probability_up, "probability_up")
        validate_unit_interval(self.confidence, "confidence")

        if self.interval is not None:
            if not isinstance(self.interval, PredictionInterval):
                raise TypeError("interval debe ser PredictionInterval")
            validate_interval_contains(self.interval, self.expected_return)

        if self.regime is not None and not isinstance(self.regime, Regime):
            raise TypeError("regime debe ser Regime")
        if self.input_context is not None and not isinstance(
            self.input_context, InputContext
        ):
            raise TypeError("input_context debe ser InputContext")
        if self.strategy is not None:
            strategy = self.strategy.strip()
            if not strategy:
                raise ValueError("strategy no puede ser vacío")
            object.__setattr__(self, "strategy", strategy)

        validate_state_consistency(
            status=self.status,
            outcome=self.outcome,
            evaluation=self.evaluation,
            reward=self.reward,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
        object.__setattr__(self, "prediction_id", prediction_id)