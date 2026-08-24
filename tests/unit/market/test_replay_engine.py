"""Tests del Market Replay (FASE 5).

Cubre el contrato temporal: reloj monótono, ventana solo con velas
cerradas, el **test de oro anti-look-ahead** (feed completo vs feed
cortado producen predicciones idénticas), invalidation honesta al
agotar el feed, y la agregación del PerformanceReport.
"""

from __future__ import annotations

import pytest
from iot_machine_learning.domain.entities.market import Candle, DataStatus
from iot_machine_learning.domain.entities.market.prediction import PredictionStatus
from iot_machine_learning.domain.entities.market.replay import (
    ClockRollbackError,
    FeatureWindow,
    HorizonStat,
    MarketReplayEngine,
    PerformanceReport,
    ReplayClock,
    ReplayEngineConfig,
    ReplayRunResult,
)
from iot_machine_learning.domain.entities.market.replay.feed import HistoricalFeed


def _candle(ts: int, close: float, interval: int = 60, *, symbol: str = "NVDA") -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=float(ts),
        data_status=DataStatus.REPLAY,
        source_provider="replay-test",
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=100.0,
        interval_seconds=interval,
    )


class SyntheticFeed:
    """Feed de test: velas 1m generadas con una onda determinista."""

    def __init__(self, n: int, *, close0: float = 100.0, step: float = 0.01) -> None:
        self.symbol = "NVDA"
        self.resolution_seconds = 60
        self._candles = tuple(
            _candle(ts, round(close0 + ts * step, 4))
            for ts in range(60, (n + 1) * 60, 60)
        )

    def iter_events(self):
        yield from self._candles


def _run(feed: HistoricalFeed, horizons: tuple[int, ...] = (60, 300)) -> ReplayRunResult:
    engine = MarketReplayEngine(
        ReplayEngineConfig(
            symbol="NVDA",
            feed=feed,
            interval_seconds=60,
            horizons_seconds=horizons,
            strategy="baseline",
        )
    )
    return engine.run()


class TestReplayClock:
    def test_advance_is_monotonic(self) -> None:
        clock = ReplayClock(now=10.0)
        clock = clock.advance_to(10.5)
        assert clock.now == 10.5

    def test_rollback_rejected(self) -> None:
        clock = ReplayClock(now=10.0)
        with pytest.raises(ClockRollbackError):
            clock.advance_to(9.9)

    def test_same_timestamp_allowed(self) -> None:
        assert ReplayClock(now=10.0).advance_to(10.0).now == 10.0


class TestFeatureWindow:
    def test_window_only_accepts_ordered_candles(self) -> None:
        window = FeatureWindow(symbol="NVDA")
        window = window.append_closed(_candle(60, 100.0))
        window = window.append_closed(_candle(120, 101.0))
        with pytest.raises(ValueError):
            window.append_closed(_candle(60, 99.0))

    def test_returns_and_stats_deterministic(self) -> None:
        window = FeatureWindow(symbol="NVDA")
        for ts in range(60, 60 * 22, 60):
            window = window.append_closed(_candle(ts, 100.0 + ts * 0.001))
        r1 = window.returns(20)
        r2 = window.returns(20)
        assert r1 == r2
        assert window.mean_return(20) > 0
        assert window.std_return(20) >= 0
        assert window.vwap(20) > 0
        assert window.typical_range(20) > 0

    def test_last_closed_respects_deadline(self) -> None:
        window = FeatureWindow(symbol="NVDA")
        for ts in (60, 120, 180):
            window = window.append_closed(_candle(ts, float(ts)))
        assert window.last_close(at_or_before=120.0) == 120.0
        assert window.last_close(at_or_before=60.0) == 60.0
        assert window.last_close(at_or_before=59.0) is None


class TestReplayEngine:
    def test_basic_run_materializes_rewards(self) -> None:
        result = _run(SyntheticFeed(n=300))
        assert result.predictions
        assert len(result.outcomes) > 0
        assert all(
            p.status == PredictionStatus.REWARDED
            for p in result.predictions
            if p.outcome is not None
        )
        for pred in result.predictions:
            assert 0.05 <= pred.probability_up <= 0.95
            assert pred.input_context is not None
            assert pred.input_context.feature_count > 0

    def test_rewarded_predictions_have_reward(self) -> None:
        result = _run(SyntheticFeed(n=300))
        rewarded = [p for p in result.predictions if p.outcome is not None]
        assert rewarded
        assert all(p.reward is not None for p in rewarded)

    def test_warmup_invisible_in_replay(self) -> None:
        """Antes de cerrar 21 velas no hay predicciones (warmup honesto)."""
        for n in (21, 22, 30):
            result = _run(SyntheticFeed(n=n))
            expected = max(0, n - 21)  # primer cierre predictivo en la vela 22
            assert len(result.predictions) == expected * 2, n

    def test_invalidation_when_feed_ends(self) -> None:
        """Los horizontes no vencidos al agotarse el feed se invalidan."""
        result = _run(SyntheticFeed(n=23), horizons=(60, 300))
        assert result.invalidated
        assert all(p.outcome is None for p in result.invalidated)

    def test_clock_monotonic_guaranteed(self) -> None:
        class BackwardsFeed:
            symbol = "NVDA"
            resolution_seconds = 60

            def __init__(self) -> None:
                self._sent = False

            def iter_events(self):
                if not self._sent:
                    self._sent = True
                    yield _candle(120, 101.0)
                yield _candle(60, 100.0)

        with pytest.raises(ClockRollbackError):
            _run(BackwardsFeed())

    def test_feed_delivers_foreign_symbol_rejected(self) -> None:
        class WrongSymbolFeed:
            symbol = "AAPL"
            resolution_seconds = 60

            def iter_events(self):
                yield _candle(60, 100.0, symbol="AAPL")

        with pytest.raises(ValueError, match="símbolo"):
            _run(WrongSymbolFeed())


class TestGoldenAntiLookAhead:
    """El test de oro: cortar el futuro no cambia el pasado."""

    _CORE_FIELDS = (
        "probability_up",
        "expected_return",
        "entry_price",
        "timestamp",
        "interval",
    )

    @staticmethod
    def _core(pred) -> tuple:
        return tuple(getattr(pred, f) for f in TestGoldenAntiLookAhead._CORE_FIELDS)

    def test_full_feed_equals_cut_feed_for_overlap(self) -> None:
        """Feed completo (con datos del futuro) vs feed cortado en el
        presente: toda predicción emitida antes del corte debe ser
        idéntica — el futuro jamás influye en el pasado."""
        full = _run(SyntheticFeed(n=300))
        cut = _run(SyntheticFeed(n=24))
        full_overlap = [
            p for p in full.predictions if p.timestamp <= cut.predictions[-1].timestamp
        ]
        assert full_overlap, "deben existir predicciones en la intersección"
        assert len(full_overlap) >= len(cut.predictions)
        for full_pred, cut_pred in zip(full_overlap, cut.predictions, strict=False):
            assert self._core(full_pred) == self._core(cut_pred)

    def test_future_price_does_not_leak_backwards(self) -> None:
        """El mismo presente con un futuro distinto (ataque directo al
        look-ahead): se reemplaza el cierre del minuto siguiente por un
        valor absurdo; ninguna predicción anterior puede cambiar."""

        class MutatedFeed(SyntheticFeed):
            def iter_events(self):
                for i, candle in enumerate(self._candles):
                    if i == 23:
                        yield _candle(int(candle.timestamp), close=999.0)
                    else:
                        yield candle

        base = _run(SyntheticFeed(n=300))
        mutated = _run(MutatedFeed(n=300))
        base_preds = {p.prediction_id: p for p in base.predictions}
        # La vela mutada (ts 1380) cierra a las 1440: solo las predicciones
        # emitidas antes de ese cierre pueden compararse (las posteriores
        # legítimamente ven la vela mutada).
        for pred in mutated.predictions:
            if pred.timestamp >= 1440.0:
                continue
            if pred.prediction_id in base_preds:
                assert self._core(pred) == self._core(base_preds[pred.prediction_id])

    def test_outcome_uses_last_close_before_now(self) -> None:
        result = _run(SyntheticFeed(n=40), horizons=(60,))
        outcomes = [p.outcome for p in result.predictions if p.outcome is not None]
        assert outcomes
        sample = outcomes[0]
        expected_horizon = 60
        assert sample.horizon_seconds == expected_horizon
        assert sample.measured_at >= sample.observation_timestamp + expected_horizon


class TestPerformanceReport:
    def test_report_counts_and_rates(self) -> None:
        result = _run(SyntheticFeed(n=300))
        report = PerformanceReport.from_run(
            "NVDA", 60, result.predictions
        )
        assert report.total_predictions == len(result.predictions)
        assert report.total_evaluated == len(result.outcomes)
        assert report.total_invalidated == len(result.invalidated)
        assert report.total_evaluated + report.total_invalidated == report.total_predictions
        for stat in report.stats:
            assert stat.predictions == stat.evaluated + stat.invalidated
            assert 0.0 <= stat.direction_rate <= 1.0
            assert 0.0 <= stat.calibration <= 1.0

    def test_report_grouped_by_horizon_and_strategy(self) -> None:
        result = _run(SyntheticFeed(n=100), horizons=(60, 300))
        report = PerformanceReport.from_run("NVDA", 60, result.predictions)
        horizons = {s.horizon_seconds for s in report.stats}
        assert horizons == {60, 300}
        assert all(s.strategy == "baseline" for s in report.stats)

    def test_merge_keeps_consistency(self) -> None:
        a = HorizonStat(horizon_seconds=60, strategy=None, predictions=2, evaluated=2,
                        direction_rate=1.0, calibration=0.1, avg_return_error=0.01, reward=1.0)
        b = HorizonStat(horizon_seconds=60, strategy=None, predictions=2, evaluated=2,
                        direction_rate=0.0, calibration=0.3, avg_return_error=0.03, reward=-1.0)
        merged = a.merge(b)
        assert merged.evaluated == 4
        assert merged.direction_rate == 0.5
        assert merged.calibration == 0.2
        assert merged.avg_return_error == 0.02
        assert merged.reward == 0.0

    def test_render_ascii_presents_scoreboard(self) -> None:
        result = _run(SyntheticFeed(n=120), horizons=(60, 300))
        report = PerformanceReport.from_run("NVDA", 60, result.predictions)
        text = report.render_ascii()
        assert "ZENIN MARKET RUN" in text
        assert "NVDA" in text
        assert "1m" in text and "5m" in text
        assert "Direction:" in text and "Reward:" in text
