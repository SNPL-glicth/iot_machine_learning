"""Pruebas unitarias de FASE 9.1: ventanas walk-forward, régimen,
modelo compuesto ponderado y reporte. Sin MySQL: todo es puro."""

from __future__ import annotations

import pytest
from iot_machine_learning.domain.entities.market import Candle, DataStatus
from iot_machine_learning.domain.entities.market.adaptation import ExpertScore
from iot_machine_learning.domain.entities.market.costs import (
    EDGE_COST_NEGATIVE,
    CostModel,
)
from iot_machine_learning.domain.entities.market.replay import (
    EdgeMetrics,
    HorizonEval,
    WfRow,
    evaluate_window,
    render_wf_report,
    weighted_model_metrics,
    wf_windows,
    window_regime,
)


def _candle(ts: float, close: float = 100.0) -> Candle:
    return Candle(
        symbol="TEST",
        timestamp=ts,
        data_status=DataStatus.REPLAY,
        source_provider="csv_replay",
        venue="NASDAQ",
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1000,
        interval_seconds=3600,
        vwap=close,
        trade_count=10,
        adjusted=False,
    )


def _candles(
    n: int, *, start: float = 1_000_000, step: float = 3600.0, closes: list[float] | None = None
) -> tuple[Candle, ...]:
    return tuple(_candle(start + i * step, closes[i] if closes else 100.0) for i in range(n))


def _score(expert, regime, horizon, n=20, hits=12, reward=8.0, cal=0.1):
    return ExpertScore(
        expert=expert,
        regime=regime,
        horizon_seconds=horizon,
        n=n,
        accuracy=hits / n,
        mean_reward=reward / n,
        reward_total=reward,
        calibration_error=cal,
        reward_adjusted=(reward / n) * (1 - 0.5 * min(cal, 1.0)),
        history_days=2,
    )


def _edge_score(expert, regime, horizon, *, expected, realized, n=20):
    return ExpertScore(
        expert=expert,
        regime=regime,
        horizon_seconds=horizon,
        n=n,
        accuracy=0.5,
        mean_reward=expected,
        reward_total=expected * n,
        calibration_error=0.0,
        reward_adjusted=expected,
        history_days=2,
        expected_return=expected,
        realized_return=realized,
        execution_costs=0.0,
    )


class TestWfWindows:
    def test_contiguous_disjoint_train_test(self):
        candles = _candles(24 * 24, start=1_000_000)  # 24 días 1h
        windows = wf_windows(
            candles,
            train_seconds=14 * 86400,
            test_seconds=7 * 86400,
            step_seconds=7 * 86400,
        )
        assert len(windows) >= 1
        w = windows[0]
        assert w.train[-1].timestamp < w.test[0].timestamp  # disjunto y ordenado
        assert w.train_start == 1_000_000
        assert w.test_start == w.train_end

    def test_origin_rolls(self):
        candles = _candles(35 * 24, start=1_000_000)
        windows = wf_windows(
            candles,
            train_seconds=14 * 86400,
            test_seconds=7 * 86400,
            step_seconds=7 * 86400,
        )
        assert len(windows) >= 2
        assert windows[1].train_start > windows[0].train_start
        # Cada ventana es contigua y disjunta: TEST sigue a su TRAIN.
        for w in windows:
            assert w.train[-1].timestamp < w.test[0].timestamp
            assert w.test_start == w.train_end
        # Los TEST avanzan en el tiempo (origin rolling).
        assert windows[1].test_start > windows[0].test_start
        # Sin fugas hacia el futuro: el TRAIN de cada ventana termina
        # exactamente donde empieza su TEST (lo aprendido llega hasta t).

    def test_min_train_honored(self):
        candles = _candles(100)
        windows = wf_windows(
            candles,
            train_seconds=86400,
            test_seconds=86400,
            step_seconds=86400,
            min_train=200,
        )
        assert windows == ()

    def test_empty(self):
        assert wf_windows((), train_seconds=1, test_seconds=1, step_seconds=1) == ()


class TestWindowRegime:
    def test_labels(self):
        flat = _candles(60, closes=[100.0 + i * 0.01 for i in range(60)])
        assert window_regime(flat) in {"bull", "bear", "neutral", "high_volatility"}

    def test_too_short_returns_none(self):
        assert window_regime(_candles(5)) is None


class TestWeightedModelMetrics:
    def test_weighted_reward(self):
        m = weighted_model_metrics(
            {"momentum": 0.7, "naive": 0.3},
            {"momentum": 0.02, "naive": -0.01},
            {"momentum": 0.6, "naive": 0.5},
        )
        assert m.model_reward == pytest.approx(0.7 * 0.02 + 0.3 * -0.01)
        assert m.model_accuracy == pytest.approx(0.7 * 0.6 + 0.3 * 0.5)

    def test_zero_weight_excluded(self):
        m = weighted_model_metrics(
            {"momentum": 1.0, "naive": 0.0},
            {"momentum": 0.02, "naive": -99.0},
            {"momentum": 0.6, "naive": 0.0},
        )
        assert m.model_reward == pytest.approx(0.02)
        assert m.model_accuracy == pytest.approx(0.6)

    def test_missing_expert_skipped(self):
        m = weighted_model_metrics(
            {"momentum": 0.5, "naive": 0.5},
            {"momentum": 0.02},
            {"momentum": 0.6},
        )
        assert m.model_reward == pytest.approx(0.02)

    def test_no_weights(self):
        m = weighted_model_metrics({}, {}, {})
        assert m.model_reward == 0.0 and m.model_accuracy == 0.0


class TestEvaluateWindow:
    WEIGHTS = {
        "*|bear|3600s": {"momentum": 0.7, "naive": 0.3},
        "*|-|3600s": {"momentum": 0.5, "naive": 0.5},
    }

    def test_scopes_to_window_regime(self):
        scores = (
            _score("momentum", "bear", 3600, reward=8.0),
            _score("naive", "bear", 3600, reward=-4.0),
            _score("momentum", "neutral", 3600, reward=99.0),  # fuera de régimen
            _score("naive", "neutral", 3600, reward=99.0),
        )
        evals = evaluate_window(scores, self.WEIGHTS, regime="bear")
        assert len(evals) == 1
        assert evals[0].n == 40
        w = self.WEIGHTS["*|bear|3600s"]
        assert evals[0].model.model_reward == pytest.approx(
            w["momentum"] * 8.0 / 20 + w["naive"] * -4.0 / 20
        )

    def test_falls_back_to_global_context(self):
        scores = (_score("momentum", "bear", 3600, reward=8.0),)
        evals = evaluate_window(scores, {"*|-|3600s": {"momentum": 1.0}}, regime="bear")
        assert evals and evals[0].model.model_reward == pytest.approx(8.0 / 20)

    def test_no_weights_no_eval(self):
        scores = (_score("momentum", "bear", 3600),)
        assert evaluate_window(scores, {}, regime="bear") == ()


class TestEdgeMetrics:
    WEIGHTS = {
        "*|bear|3600s": {"momentum": 0.7, "naive": 0.3},
        "*|-|3600s": {"momentum": 0.5, "naive": 0.5},
    }
    COSTS = CostModel(spread_bps=4.0, slippage_bps=5.0, commission_bps=3.0)  # 12bps

    def test_edge_present_only_with_cost_model(self):
        scores = (
            _edge_score("momentum", "bear", 3600, expected=0.0015, realized=0.0010),
            _edge_score("naive", "bear", 3600, expected=0.0005, realized=-0.0005),
        )
        bare = evaluate_window(scores, self.WEIGHTS, regime="bear")
        assert bare[0].edge is None
        with_costs = evaluate_window(scores, self.WEIGHTS, regime="bear", cost_model=self.COSTS)
        edge = with_costs[0].edge
        assert edge is not None
        assert edge.cost_bps == 12

    def test_edge_weights_and_costs(self):
        scores = (
            _edge_score("momentum", "bear", 3600, expected=0.0015, realized=0.0010),
            _edge_score("naive", "bear", 3600, expected=0.0005, realized=-0.0005),
        )
        edge = evaluate_window(scores, self.WEIGHTS, regime="bear", cost_model=self.COSTS)[0].edge
        w = self.WEIGHTS["*|bear|3600s"]
        assert edge.expected_gross == pytest.approx(w["momentum"] * 0.0015 + w["naive"] * 0.0005)
        assert edge.realized_gross == pytest.approx(w["momentum"] * 0.0010 + w["naive"] * -0.0005)
        cost = self.COSTS.total()
        assert edge.expected_net == pytest.approx(edge.expected_gross - cost)
        assert edge.realized_net == pytest.approx(edge.realized_gross - cost)

    def test_realized_net_negative_after_costs(self):
        # El caso cruel de la FASE 9.2: señal bruta positiva que muere a costos.
        scores = (
            _edge_score("momentum", "bear", 3600, expected=0.0015, realized=0.0010),
            _edge_score("naive", "bear", 3600, expected=0.0005, realized=-0.0005),
        )
        edge = evaluate_window(scores, self.WEIGHTS, regime="bear", cost_model=self.COSTS)[0].edge
        assert edge.realized_gross > 0
        assert edge.realized_net < 0
        assert edge.expected_gross > edge.expected_net


class TestRenderWfReport:
    def test_renders_rows_and_aggregate(self):
        rows = [
            WfRow(
                index=0,
                symbol="TEST",
                regime="bear",
                train_start=0.0,
                train_end=1.0,
                test_start=1.0,
                test_end=2.0,
                n_train=10,
                accepted=1,
                rejected=2,
                horizons=(
                    HorizonEval(
                        horizon_seconds=3600,
                        regime="bear",
                        n=20,
                        weights={"momentum": 0.7, "naive": 0.3},
                        experts={
                            "momentum": {
                                "accuracy": 0.6,
                                "mean_reward": 0.02,
                                "n": 20,
                                "reward_adjusted": 0.01,
                            },
                            "naive": {
                                "accuracy": 0.4,
                                "mean_reward": -0.01,
                                "n": 20,
                                "reward_adjusted": -0.01,
                            },
                        },
                        model=weighted_model_metrics(
                            {"momentum": 0.7, "naive": 0.3},
                            {"momentum": 0.02, "naive": -0.01},
                            {"momentum": 0.6, "naive": 0.4},
                        ),
                    ),
                ),
            )
        ]
        text = render_wf_report(rows, symbol="TEST", interval_label="1h")
        assert "WALK-FORWARD" in text
        assert "W00" in text
        assert "bear" in text
        assert "AGREGADO" in text
        assert "reward" in text

    def test_edge_line_when_edge_class_present(self):
        rows = [
            WfRow(
                index=0,
                symbol="TEST",
                regime="bear",
                train_start=0.0,
                train_end=1.0,
                test_start=1.0,
                test_end=2.0,
                n_train=10,
                accepted=1,
                rejected=2,
                horizons=(
                    HorizonEval(
                        horizon_seconds=3600,
                        regime="bear",
                        n=20,
                        weights={"momentum": 1.0},
                        experts={
                            "momentum": {
                                "accuracy": 0.6,
                                "mean_reward": 0.02,
                                "n": 20,
                                "reward_adjusted": 0.01,
                            },
                        },
                        model=weighted_model_metrics(
                            {"momentum": 1.0},
                            {"momentum": 0.02},
                            {"momentum": 0.6},
                        ),
                        edge=EdgeMetrics(
                            expected_gross=0.0015,
                            expected_net=0.0003,
                            realized_gross=0.0010,
                            realized_net=-0.0002,
                            cost_bps=12,
                            n=20,
                        ),
                    ),
                ),
                cost_bps=12,
                sharpe=-0.1,
                edge_class=EDGE_COST_NEGATIVE,
            )
        ]
        text = render_wf_report(rows, symbol="TEST", interval_label="1h")
        assert "EDGE (12bps)" in text
        assert EDGE_COST_NEGATIVE in text
        assert "sharpe -0.10" in text
        assert "costos 12bps" in text

    def test_empty(self):
        text = render_wf_report((), symbol="TEST", interval_label="1h")
        assert "sin ventanas" in text
