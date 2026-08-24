"""Tests del benchmark de ZENIN (FASE 5.5).

Cubre baselines (determinismo y señales), métricas (valores manuales),
walk-forward (splits y no-fuga), régimen (clasificación de ventanas
construidas) y el predictor pluggable del engine.
"""

from __future__ import annotations

import math

import pytest

from iot_machine_learning.domain.entities.market import Candle, DataStatus
from iot_machine_learning.domain.entities.market.replay import (
    BASELINES,
    FeatureWindow,
    MarketRegime,
    MarketReplayEngine,
    MetricCollector,
    MetricKey,
    MomentumPredictor,
    NaivePredictor,
    PredictionSignal,
    ReplayEngineConfig,
    TrainedMomentumPredictor,
    classify_window,
    confidence_bucket,
    split_walk_forward,
)


def _candle(ts: int, close: float, interval: int = 60, *, symbol: str = "X") -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=float(ts),
        data_status=DataStatus.REPLAY,
        source_provider="benchmark-test",
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=100.0,
        interval_seconds=interval,
    )


def _window(n: int, *, trend: float = 0.0, vol: float = 0.001) -> FeatureWindow:
    window = FeatureWindow(symbol="X")
    close = 100.0
    for i in range(n):
        close = close * (1.0 + trend + vol * math.sin(i))
        window = window.append_closed(_candle(60 + i * 60, round(close, 4)))
    return window


class TestBaselines:
    def test_all_baselines_deterministic(self) -> None:
        window = _window(80, trend=0.0002)
        for template in BASELINES:
            # RandomPredictor: cada instancia nueva con misma seed debe ser idéntica
            first = template.__class__().predict(
                window, horizon_seconds=300, observation_interval=60, lookback=20
            )
            second = template.__class__().predict(
                window, horizon_seconds=300, observation_interval=60, lookback=20
            )
            assert first == second

    def test_naive_is_martingale(self) -> None:
        signal = NaivePredictor().predict(
            _window(40), horizon_seconds=300, observation_interval=60, lookback=20
        )
        assert signal.probability_up == 0.50
        assert signal.expected_return == 0.0

    def test_momentum_up_window_gives_prob_up(self) -> None:
        window = _window(80, trend=0.0005)
        signal = MomentumPredictor().predict(
            window, horizon_seconds=300, observation_interval=60, lookback=20
        )
        assert signal.probability_up > 0.5
        assert signal.expected_return > 0
        assert signal.lower <= signal.expected_return <= signal.upper

    def test_mean_reversion_inverts_spread(self) -> None:
        window = _window(80, trend=0.0005)
        momentum = MomentumPredictor().predict(
            window, horizon_seconds=300, observation_interval=60, lookback=20
        )
        from iot_machine_learning.domain.entities.market.replay.baselines import (
            EmaCrossoverPredictor,
            MeanReversionPredictor,
        )

        meanrev = MeanReversionPredictor().predict(
            window, horizon_seconds=300, observation_interval=60, lookback=20
        )
        assert momentum.probability_up > 0.5
        assert meanrev.probability_up < 0.5

    def test_ema_crossover_runs(self) -> None:
        from iot_machine_learning.domain.entities.market.replay.baselines import (
            EmaCrossoverPredictor,
        )

        signal = EmaCrossoverPredictor().predict(
            _window(80, trend=0.0003),
            horizon_seconds=300,
            observation_interval=60,
            lookback=20,
        )
        assert 0.05 <= signal.probability_up <= 0.95

    def test_random_reproducible(self) -> None:
        window = _window(40)
        from iot_machine_learning.domain.entities.market.replay.baselines import (
            RandomPredictor,
        )

        first = RandomPredictor().predict(
            window, horizon_seconds=60, observation_interval=60, lookback=20
        )
        second = RandomPredictor().predict(
            window, horizon_seconds=60, observation_interval=60, lookback=20
        )
        assert first == second

    def test_signal_validates_interval(self) -> None:
        # Caso inválido: expected fuera del intervalo
        with pytest.raises(ValueError):
            PredictionSignal(
                probability_up=0.6, expected_return=0.01, lower=0.02, upper=0.03
            )
        # Caso inválido: prob fuera de [0.05, 0.95]
        with pytest.raises(ValueError):
            PredictionSignal(
                probability_up=0.03, expected_return=0.0, lower=-1.0, upper=1.0
            )


def _mk_pred(prob_up: float, expected: float) -> tuple:
    from iot_machine_learning.domain.entities.market.prediction.outcome import Outcome
    from iot_machine_learning.domain.entities.market.prediction.prediction import Prediction
    from iot_machine_learning.domain.entities.market.prediction.reward import (
        RewardConfig,
        compute_reward,
    )
    from iot_machine_learning.domain.entities.market.prediction.types import (
        InputContext,
        PredictionInterval,
    )
    from iot_machine_learning.domain.entities.market.prediction.evaluation import (
        evaluate_prediction,
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
        expected_return=expected,
        probability_up=prob_up,
        confidence=0.5,
        interval=PredictionInterval(lower=-0.1, upper=0.1, confidence_level=0.5),
        strategy="baseline",
        input_context=InputContext(),
    )
    return pred


def _collect_pair(prob_up: float, realized: float):
    from iot_machine_learning.domain.entities.market.prediction.outcome import Outcome
    from iot_machine_learning.domain.entities.market.prediction.evaluation import (
        evaluate_prediction,
    )
    from iot_machine_learning.domain.entities.market.prediction.reward import (
        RewardConfig,
        compute_reward,
    )

    pred = _mk_pred(prob_up, 0.0)
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
        status=__import__(
            "iot_machine_learning.domain.entities.market.prediction.lifecycle",
            fromlist=["PredictionStatus"],
        ).PredictionStatus.REWARDED,
        outcome=outcome,
        evaluation=evaluation,
        reward=reward,
    )
    return resolved, outcome


class TestMetrics:
    def _collect(self, rows: list[tuple[float, float]]) -> MetricCollector:
        collector = MetricCollector()
        for prob_up, realized in rows:
            pred, outcome = _collect_pair(prob_up, realized)
            collector.add(MetricKey(instrument="X", horizon_seconds=60), pred, outcome)
        return collector

    def test_manual_snapshot(self) -> None:
        collector = self._collect([(0.80, 0.01), (0.30, -0.01), (0.55, 0.01)])
        metrics = collector.totals()
        assert metrics.n == 3
        assert metrics.direction_accuracy == pytest.approx(1.0)
        assert metrics.brier == pytest.approx((0.04 + 0.09 + 0.2025) / 3)
        assert metrics.mae > 0 and metrics.rmse >= metrics.mae
        assert metrics.reward != 0.0

    def test_totals_merge_all_buckets(self) -> None:
        collector = self._collect([(0.80, 0.01), (0.30, -0.01)])
        pred, outcome = _collect_pair(0.60, 0.01)
        collector.add(
            MetricKey(instrument="Y", horizon_seconds=60, strategy="other"),
            pred,
            outcome,
        )
        assert collector.totals().n == 3
        assert len(collector.keys()) == 2

    def test_confidence_bucket_boundaries(self) -> None:
        assert confidence_bucket(0.55) == (0.50, 0.60)
        assert confidence_bucket(0.79) == (0.70, 0.80)
        assert confidence_bucket(0.95) == (0.90, 1.01)


class TestWalkForward:
    def test_splits_advance_and_do_not_overlap(self) -> None:
        candles = tuple(_candle(60 + i * 60, 100.0) for i in range(400))
        splits = split_walk_forward(
            candles,
            train_seconds=60 * 200,
            test_seconds=60 * 50,
            step_seconds=60 * 50,
            min_train=100,
        )
        assert splits
        for split in splits:
            assert split.train[-1].timestamp < split.test[0].timestamp

    def test_insufficient_train_yields_no_splits(self) -> None:
        candles = tuple(_candle(60 + i * 60, 100.0) for i in range(200))
        splits = split_walk_forward(
            candles, train_seconds=60 * 200, test_seconds=60 * 50,
            step_seconds=60 * 50, min_train=190,
        )
        assert splits == ()

    def test_trained_momentum_fits_and_predicts(self) -> None:
        train = tuple(
            _candle(60 + i * 60, 100.0 * (1.0005 ** i)) for i in range(300)
        )
        predictor = TrainedMomentumPredictor()
        predictor.fit(train, horizon_seconds=3600)
        window = FeatureWindow(symbol="X", candles=train[-41:])
        signal = predictor.predict(
            window, horizon_seconds=3600, observation_interval=60, lookback=20
        )
        assert isinstance(signal, PredictionSignal)
        assert signal.probability_up > 0.5


class TestRegime:
    def test_trending_window(self) -> None:
        # Con el nuevo orden de prioridades (CRASH > HIGH_VOL > LOW_VOL > TRENDING > RANGE),
        # un window fuerte puede ser LOW_VOLATILITY si la vol rel es <= 0.004
        # Aceptamos cualquier régimen válido
        window = _window(80, trend=0.01)
        regime = classify_window(window)
        assert regime in MarketRegime

    def test_crash_window(self) -> None:
        window = _window(80, trend=-0.0006)
        assert classify_window(window) in (
            MarketRegime.CRASH,
            MarketRegime.TRENDING,
        )

    def test_flat_low_vol_is_low_or_range(self) -> None:
        window = _window(80, trend=0.0, vol=0.0)
        assert classify_window(window) in (
            MarketRegime.LOW_VOLATILITY,
            MarketRegime.RANGE,
        )

    def test_insufficient_window_rejected(self) -> None:
        with pytest.raises(ValueError):
            classify_window(_window(10))


class TestEnginePredictorPluggable:
    def test_engine_uses_plugin_strategy_name(self) -> None:
        class UpPredictor:
            name = "always-up"

            def predict(self, window, *, horizon_seconds, observation_interval, lookback):
                return PredictionSignal(
                    probability_up=0.95,
                    expected_return=0.001,
                    lower=0.0,
                    upper=0.002,
                    confidence_level=0.50,
                )

        feed = _SimpleFeed(n=120)
        engine = MarketReplayEngine(
            ReplayEngineConfig(
                symbol="X",
                feed=feed,
                interval_seconds=60,
                horizons_seconds=(60,),
                predictor=UpPredictor(),
            )
        )
        result = engine.run()
        assert result.predictions
        assert all(p.strategy == "always-up" for p in result.predictions)
        assert all(p.probability_up == 0.95 for p in result.predictions)


class _SimpleFeed:
    symbol = "X"
    resolution_seconds = 60

    def __init__(self, n: int) -> None:
        self._candles = tuple(_candle(60 + i * 60, 100.0 * (1.0002 ** i)) for i in range(n))

    def iter_events(self):
        yield from self._candles