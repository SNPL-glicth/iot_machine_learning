"""Mapeo fila <-> entidad del MarketPredictionRepository (FASE 7).

Sin MySQL: verifican que el round trip row <-> Prediction (y el snapshot
JSON de la observación) conservan el contenido, en cada etapa del ciclo
de vida (PENDING, WAITING_OUTCOME, REWARDED, INVALIDATED).
"""

from __future__ import annotations

import json

import pytest
from iot_machine_learning.domain.entities.market import (
    Candle,
    DataStatus,
    OrderBookSnapshot,
    Quote,
    Trade,
)
from iot_machine_learning.domain.entities.market.prediction import (
    Outcome,
    Prediction,
    PredictionInterval,
    PredictionStatus,
    Regime,
    RewardConfig,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.market_prediction_repository import (
    observation_from_json,
    observation_to_json,
    prediction_to_row,
    row_to_prediction,
)

TS = 1_500_000.0


def _candle(timestamp: float = TS) -> Candle:
    return Candle(
        symbol="NVDA",
        timestamp=timestamp,
        data_status=DataStatus.REPLAY,
        source_provider="csv_replay",
        venue="NASDAQ",
        open=99.5,
        high=101.2,
        low=99.0,
        close=100.4,
        volume=12_345,
        interval_seconds=60,
        vwap=100.1,
        trade_count=120,
        adjusted=False,
    )


def _prediction(timestamp: float = TS) -> Prediction:
    return Prediction(
        prediction_id=f"NVDA-{int(timestamp)}-900",
        observation=_candle(timestamp - 60),
        horizon_seconds=900,
        timestamp=timestamp,
        entry_price=100.4,
        expected_return=0.018,
        probability_up=0.71,
        confidence=0.68,
        interval=PredictionInterval(lower=-0.012, upper=0.045, confidence_level=0.90),
        regime=Regime.HIGH_VOLATILITY,
        strategy="zenin_v1",
    )


def _rewarded(prediction: Prediction) -> Prediction:
    outcome = Outcome.from_prices(
        symbol="NVDA",
        ref_timestamp=prediction.observation.timestamp,
        ref_price=prediction.entry_price,
        horizon_seconds=prediction.horizon_seconds,
        final_price=101.58,
    )
    return (
        prediction.activate()
        .to_waiting_outcome(outcome)
        .evaluate(outcome)
        .issue_reward(RewardConfig())
    )


class TestObservationSnapshot:
    def test_candle_round_trip(self) -> None:
        original = _candle()
        restored = observation_from_json(observation_to_json(original))
        assert restored == original
        assert type(restored) is Candle

    def test_trade_round_trip(self) -> None:
        original = Trade(
            symbol="NVDA",
            timestamp=TS,
            data_status=DataStatus.REALTIME,
            source_provider="alpaca",
            venue="NASDAQ",
            price=100.42,
            size=55,
            trade_id="t-1",
            taker_side="buy",
            conditions=("T", "R"),
            tape="C",
            corrected=False,
        )
        restored = observation_from_json(observation_to_json(original))
        assert restored == original
        assert restored.conditions == ("T", "R")

    def test_quote_round_trip(self) -> None:
        original = Quote(
            symbol="NVDA",
            timestamp=TS,
            data_status=DataStatus.DELAYED,
            source_provider="alpaca",
            venue="NASDAQ",
            bid=100.40,
            bid_size=10,
            ask=100.45,
            ask_size=15,
            bid_exchange="Q",
            ask_exchange="Q",
            conditions=("R",),
            tape="C",
        )
        assert observation_from_json(observation_to_json(original)) == original

    def test_order_book_round_trip(self) -> None:
        original = OrderBookSnapshot(
            symbol="NVDA",
            timestamp=TS,
            data_status=DataStatus.REALTIME,
            source_provider="alpaca",
            venue="NASDAQ",
            bids=((100.40, 10), (100.35, 8)),
            asks=((100.45, 12), (100.50, 20)),
            reset=True,
        )
        restored = observation_from_json(observation_to_json(original))
        assert restored == original
        assert restored.bids == ((100.40, 10), (100.35, 8))

    def test_unsupported_type_rejected(self) -> None:
        with pytest.raises(TypeError, match="MarketObservation"):
            observation_to_json(object())  # type: ignore[arg-type]

    def test_snapshot_is_compact_json(self) -> None:
        payload = observation_to_json(_candle())
        assert isinstance(json.loads(payload), dict)
        assert payload.count(" ") < 5


class TestPredictionRoundTrip:
    def test_pending_round_trip(self) -> None:
        original = _prediction()
        restored = row_to_prediction(prediction_to_row(original))
        assert restored.prediction_id == original.prediction_id
        assert restored.observation == original.observation
        assert restored.horizon_seconds == original.horizon_seconds
        assert restored.entry_price == original.entry_price
        assert restored.expected_return == original.expected_return
        assert restored.probability_up == original.probability_up
        assert restored.confidence == original.confidence
        assert restored.interval == original.interval
        assert restored.regime == original.regime
        assert restored.strategy == original.strategy
        assert restored.input_context == original.input_context
        assert restored.status is PredictionStatus.PENDING
        assert restored.outcome is None
        assert restored.reward is None

    def test_rewarded_round_trip(self) -> None:
        original = _rewarded(_prediction())
        restored = row_to_prediction(prediction_to_row(original))
        assert restored.status is PredictionStatus.REWARDED
        assert restored.outcome is not None
        assert restored.outcome == original.outcome
        assert restored.evaluation is not None
        assert restored.evaluation == original.evaluation
        assert restored.reward is not None
        assert restored.reward == original.reward

    def test_invalidated_round_trip(self) -> None:
        original = _prediction().invalidate("provider_gap")
        restored = row_to_prediction(prediction_to_row(original))
        assert restored.status is PredictionStatus.INVALIDATED
        assert restored.invalidation_reason == "provider_gap"
        assert restored.reward is None

    def test_reward_total_is_positive_when_correct(self) -> None:
        row = prediction_to_row(_rewarded(_prediction()))
        assert row["reward_total"] > 0
        assert row["direction_correct"] is True
        assert row["outcome_final_price"] == 101.58
