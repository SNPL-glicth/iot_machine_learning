"""Tests de MetricCollector.metrics(key) — filtrado por clave sin regresión en totals()."""

from __future__ import annotations

from iot_machine_learning.domain.entities.market import Candle, DataStatus
from iot_machine_learning.domain.entities.market.replay import (
    MetricCollector,
    MetricKey,
)
from iot_machine_learning.domain.entities.market.replay.metrics import (
    _EmptyMetrics,
)


def _candle(ts: int, close: float) -> Candle:
    return Candle(
        symbol="X",
        timestamp=float(ts),
        data_status=DataStatus.REPLAY,
        source_provider="metrics-key-test",
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=100.0,
        interval_seconds=60,
    )


def _pair(prob_up: float, realized: float):
    from iot_machine_learning.domain.entities.market.prediction.evaluation import (
        evaluate_prediction,
    )
    from iot_machine_learning.domain.entities.market.prediction.outcome import Outcome
    from iot_machine_learning.domain.entities.market.prediction.prediction import (
        Prediction,
    )
    from iot_machine_learning.domain.entities.market.prediction.reward import (
        RewardConfig,
        compute_reward,
    )
    from iot_machine_learning.domain.entities.market.prediction.types import (
        InputContext,
        PredictionInterval,
    )
    from iot_machine_learning.domain.entities.market.replay.engine import (
        PredictionStatus,
    )

    obs = _candle(60, 100.0)
    pred = Prediction(
        prediction_id="p",
        observation=obs,
        horizon_seconds=60,
        timestamp=60.0,
        entry_price=100.0,
        expected_return=0.0,
        probability_up=prob_up,
        confidence=0.5,
        interval=PredictionInterval(lower=-0.1, upper=0.1, confidence_level=0.5),
        strategy="baseline",
        input_context=InputContext(),
    )
    outcome = Outcome.from_prices(
        symbol="X",
        ref_timestamp=60.0,
        ref_price=100.0,
        horizon_seconds=60,
        final_price=100.0 * (1.0 + realized),
        measured_at=120.0,
    )
    evaluation = evaluate_prediction(pred, outcome)
    reward = compute_reward(pred, outcome, evaluation, RewardConfig())
    resolved = pred.__class__(
        prediction_id=pred.prediction_id,
        observation=pred.observation,
        horizon_seconds=pred.horizon_seconds,
        timestamp=pred.timestamp,
        entry_price=pred.entry_price,
        expected_return=pred.expected_return,
        probability_up=pred.probability_up,
        confidence=pred.confidence,
        interval=pred.interval,
        strategy=pred.strategy,
        input_context=pred.input_context,
        status=PredictionStatus.REWARDED,
        outcome=outcome,
        evaluation=evaluation,
        reward=reward,
    )
    return resolved, outcome


class TestMetricsByKey:
    def test_metrics_none_equals_totals(self) -> None:
        collector = MetricCollector()
        for prob_up, realized in [(0.80, 0.01), (0.30, -0.01)]:
            pred, outcome = _pair(prob_up, realized)
            collector.add(MetricKey(instrument="X", horizon_seconds=60), pred, outcome)
        assert collector.metrics().n == collector.totals().n == 2

    def test_metrics_filters_by_key(self) -> None:
        collector = MetricCollector()
        k60 = MetricKey(instrument="X", horizon_seconds=60)
        k3600 = MetricKey(instrument="X", horizon_seconds=3600)
        pred, outcome = _pair(0.80, 0.01)
        collector.add(k60, pred, outcome)
        pred2, outcome2 = _pair(0.30, -0.01)
        collector.add(k3600, pred2, outcome2)

        assert collector.metrics(k60).n == 1
        assert collector.metrics(k3600).n == 1
        assert collector.totals().n == 2

    def test_metrics_unknown_key_empty(self) -> None:
        collector = MetricCollector()
        pred, outcome = _pair(0.80, 0.01)
        collector.add(MetricKey(instrument="X", horizon_seconds=60), pred, outcome)
        missing = collector.metrics(MetricKey(instrument="ZZZ", horizon_seconds=60))
        assert isinstance(missing, _EmptyMetrics)
        assert missing.n == 0
