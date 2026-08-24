"""OutcomeResolver (FASE 7) — tests unitarios del dominio puro.

Sin MySQL ni feeds reales: precios simulados vía ``PriceLookup``.
"""

from __future__ import annotations

import pytest
from iot_machine_learning.domain.entities.market import Candle, DataStatus
from iot_machine_learning.domain.entities.market.prediction import (
    Outcome,
    Prediction,
    PredictionInterval,
    PredictionStatus,
    Regime,
)
from iot_machine_learning.domain.entities.market.prediction.resolver import (
    OutcomeResolver,
    PriceLookup,
    ResolvedBatch,
)

TS0 = 1_000_000.0
DEADLINE = TS0 - 60 + 900  # observación en TS0-60 + horizonte 900s


def _candle(timestamp: float) -> Candle:
    return Candle(
        symbol="NVDA",
        timestamp=timestamp,
        data_status=DataStatus.REPLAY,
        source_provider="csv_replay",
        venue="NASDAQ",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000,
        interval_seconds=60,
        vwap=100.0,
        trade_count=10,
        adjusted=False,
    )


def _prediction(
    *,
    timestamp: float = TS0,
    horizon: int = 900,
    price: float = 100.0,
    status: PredictionStatus = PredictionStatus.PENDING,
    prediction_id: str = "NVDA-1000000-900",
) -> Prediction:
    return Prediction(
        prediction_id=prediction_id,
        observation=_candle(timestamp - 60),
        horizon_seconds=horizon,
        timestamp=timestamp,
        entry_price=price,
        expected_return=0.02,
        probability_up=0.70,
        confidence=0.60,
        interval=PredictionInterval(lower=-0.01, upper=0.05, confidence_level=0.90),
        regime=Regime.BULL,
        strategy="zenin_v1",
        status=status,
    )


class _Prices(PriceLookup):
    """Precios simulados: último cierre disponible en/antes del plazo."""

    def __init__(self, closes: dict[float, float] | None = None) -> None:
        self.closes = closes or {}

    def last_close(self, at_or_before: float) -> float | None:
        available = [t for t in self.closes if t <= at_or_before]
        if not available:
            return None
        return self.closes[max(available)]


class TestOutcomeResolver:
    def test_resolve_when_horizon_elapsed(self) -> None:
        """Horizonte vencido + precio disponible → ciclo completo a REWARDED."""
        resolver = OutcomeResolver()
        pred = _prediction()
        prices = _Prices({DEADLINE: 102.5})
        batch = resolver.resolve([pred], prices)
        assert isinstance(batch, ResolvedBatch)
        assert batch.resolved_count == 1
        assert batch.waiting_count == 0
        resolved = batch.resolved[0]
        assert resolved.status is PredictionStatus.REWARDED
        assert resolved.outcome is not None
        assert resolved.outcome.final_price == 102.5
        assert resolved.outcome.return_realized == pytest.approx(0.025)
        assert resolved.evaluation is not None
        assert resolved.evaluation.direction_correct is True
        assert resolved.reward is not None
        assert resolved.reward.total > 0

    def test_still_waiting_when_no_price_yet(self) -> None:
        """Sin precio en/antes del vencimiento → sigue en espera, intacta."""
        resolver = OutcomeResolver()
        pred = _prediction()
        batch = resolver.resolve([pred], _Prices({}))
        assert batch.resolved_count == 0
        assert batch.waiting_count == 1
        assert batch.still_waiting[0] is pred

    def test_resolve_from_active_and_waiting_states(self) -> None:
        """ACTIVE y WAITING_OUTCOME también se resuelven (re-runs parciales)."""
        resolver = OutcomeResolver()
        active = _prediction(prediction_id="NVDA-1000000-901", status=PredictionStatus.ACTIVE)
        waiting = _prediction(prediction_id="NVDA-1000000-902", status=PredictionStatus.ACTIVE)

        waiting = waiting.to_waiting_outcome(
            Outcome.from_prices(
                symbol="NVDA",
                ref_timestamp=TS0 - 60,
                ref_price=100.0,
                horizon_seconds=900,
                final_price=101.0,
            )
        )
        batch = resolver.resolve([active, waiting], _Prices({DEADLINE: 102.0}))
        assert batch.resolved_count == 2
        assert all(p.status is PredictionStatus.REWARDED for p in batch.resolved)

    def test_terminal_predictions_untouched(self) -> None:
        """REWARDED e INVALIDATED jamás se re-resuelven ni mutan."""
        resolver = OutcomeResolver()
        rewarded = _prediction(prediction_id="NVDA-1000000-903")
        rewarded = rewarded.activate().to_waiting_outcome(
            Outcome.from_prices(
                symbol="NVDA",
                ref_timestamp=TS0 - 60,
                ref_price=100.0,
                horizon_seconds=900,
                final_price=101.0,
            )
        ).evaluate(
            Outcome.from_prices(
                symbol="NVDA",
                ref_timestamp=TS0 - 60,
                ref_price=100.0,
                horizon_seconds=900,
                final_price=101.0,
            )
        ).issue_reward(resolver.reward_config)
        invalidated = _prediction(
            prediction_id="NVDA-1000000-904",
            status=PredictionStatus.ACTIVE,
        ).invalidate("provider_gap")
        batch = resolver.resolve([rewarded, invalidated], _Prices({DEADLINE: 105.0}))
        assert batch.resolved_count == 0
        assert batch.waiting_count == 0
        assert batch.unchanged == (rewarded, invalidated)

    def test_invalidated_never_gets_reward(self) -> None:
        """Aunque haya precio al vencimiento, lo invalidado sigue sin reward."""
        resolver = OutcomeResolver()
        pred = _prediction(status=PredictionStatus.ACTIVE).invalidate("feed_ended")
        batch = resolver.resolve([pred], _Prices({DEADLINE: 150.0}))
        assert batch.resolved_count == 0
        assert pred.reward is None
        assert pred.status is PredictionStatus.INVALIDATED

    def test_wrong_input_type_rejected(self) -> None:
        resolver = OutcomeResolver()
        with pytest.raises(TypeError, match="Prediction"):
            resolver.resolve(["no-soy-una-prediccion"], _Prices({}))  # type: ignore[list-item]

    def test_batch_is_immutable(self) -> None:
        batch = OutcomeResolver().resolve([], _Prices({}))
        with pytest.raises(AttributeError):
            batch.resolved = ()  # type: ignore[misc]
