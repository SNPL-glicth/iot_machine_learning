"""FASE 9.3 — Matriz de ablations: pesos, portafolio y agregados (puro)."""

import pytest
from iot_machine_learning.domain.entities.market.adaptation import ExpertScore
from iot_machine_learning.domain.entities.market.replay.ablation import (
    ABLATION_EMA,
    ABLATION_FULL,
    ABLATION_MOMENTUM,
    ABLATION_NAIVE,
    ABLATION_NO_MEMORY,
    ABLATION_NO_MOE,
    ABLATION_NO_REGIME,
    AblationStats,
    AblationWindow,
    ablation_weights,
    ablation_window_stats,
    active_versions_by_window,
    aggregate_ablation,
    max_drawdown,
    parse_version_reason,
    portfolio_net_returns,
    render_ablation_matrix,
    sharpe_of,
)

WEIGHTS = {
    "*|bear|3600s": {"momentum": 0.7, "naive": 0.3, "ema-crossover": 0.0},
    "*|-|3600s": {"momentum": 0.5, "naive": 0.5},
}


def _score(expert, regime, reward_adjusted, expected=0.001):
    return ExpertScore(
        expert=expert,
        regime=regime,
        horizon_seconds=3600,
        n=20,
        accuracy=0.5,
        mean_reward=0.0,
        reward_total=0.0,
        calibration_error=0.0,
        reward_adjusted=reward_adjusted,
        history_days=2,
        expected_return=expected,
        realized_return=expected,
        execution_costs=0.0,
    )


SCORES = (
    _score("naive", "bear", 0.10),
    _score("momentum", "bear", 0.50),
    _score("ema-crossover", "bear", 0.30),
    _score("mean-reversion", "bear", 0.20),
    _score("naive", "neutral", 0.05),
)


class TestParseVersionReason:
    def test_parses_wf_reason(self):
        assert parse_version_reason("wf NVDA W04: 1 propuesta(s) aceptada(s)") == ("NVDA", 4)
        assert parse_version_reason("wf BTC-USD W102: 0") == ("BTC-USD", 102)

    def test_rejects_other_reasons(self):
        assert parse_version_reason("bootstrap FASE 8") is None
        assert parse_version_reason("") is None


class TestActiveVersionsByWindow:
    def _version(self, version_id, reason, created_at=100.0):
        return {"version_id": version_id, "reason": reason, "created_at": created_at, "weights": "{}"}

    def test_inherits_before_first_version(self):
        rows = [
            self._version(1, "bootstrap FASE 8"),
            self._version(2, "wf NVDA W05: ..."),
            self._version(3, "wf NVDA W06: ..."),
        ]
        active = active_versions_by_window(rows, "NVDA", [0, 1, 4, 5, 6])
        assert active[0]["version_id"] == 1  # heredada (v1)
        assert active[4]["version_id"] == 1
        assert active[5]["version_id"] == 2
        assert active[6]["version_id"] == 3

    def test_runs_resolved_in_version_order(self):
        # Re-corrida: la segunda corrida de NVDA re-adapta W04-W06.
        rows = [
            self._version(10, "bootstrap"),
            self._version(11, "wf NVDA W05: ..."),  # corrida 1
            self._version(12, "wf NVDA W06: ..."),  # corrida 1
            self._version(13, "wf NVDA W04: ..."),  # corrida 2
            self._version(14, "wf NVDA W05: ..."),  # corrida 2
            self._version(15, "wf NVDA W06: ..."),  # corrida 2
        ]
        active = active_versions_by_window(rows, "NVDA", [0, 1, 2, 3, 4, 5, 6])
        assert active[0]["version_id"] == 10
        assert active[3]["version_id"] == 10  # corrida 2 aún no adapta W00-W03
        assert active[4]["version_id"] == 13  # corrida 2 gana sobre corrida 1 (W05)
        assert active[5]["version_id"] == 14
        assert active[6]["version_id"] == 15

    def test_no_versions_for_symbol_uses_global_latest(self):
        rows = [self._version(1, "wf AMD W00: ..."), self._version(2, "wf AMD W01: ...")]
        active = active_versions_by_window(rows, "NVDA", [0, 1])
        assert active[0]["version_id"] == 2
        assert active[1]["version_id"] == 2

    def test_no_versions_at_all(self):
        active = active_versions_by_window([], "NVDA", [0, 1])
        assert active[0] is None


class TestAblationWeights:
    def test_baselines_single_expert(self):
        assert ablation_weights(
            ABLATION_NAIVE, weights_by_context=WEIGHTS, regime="bear", horizon=3600, scores=SCORES
        ) == {"naive": 1.0}
        assert ablation_weights(
            ABLATION_MOMENTUM, weights_by_context=WEIGHTS, regime="bear", horizon=3600, scores=SCORES
        ) == {"momentum": 1.0}
        assert ablation_weights(
            ABLATION_EMA, weights_by_context=WEIGHTS, regime="bear", horizon=3600, scores=SCORES
        ) == {"ema-crossover": 1.0}

    def test_no_memory_uniform_over_scoped(self):
        weights = ablation_weights(
            ABLATION_NO_MEMORY, weights_by_context=WEIGHTS, regime="bear", horizon=3600, scores=SCORES
        )
        assert weights == {
            "naive": 0.25,
            "momentum": 0.25,
            "ema-crossover": 0.25,
            "mean-reversion": 0.25,
        }

    def test_no_regime_uses_global_context(self):
        weights = ablation_weights(
            ABLATION_NO_REGIME, weights_by_context=WEIGHTS, regime="bear", horizon=3600, scores=SCORES
        )
        assert weights == {"momentum": 0.5, "naive": 0.5}

    def test_no_regime_falls_back_to_uniform_without_global(self):
        weights = ablation_weights(
            ABLATION_NO_REGIME,
            weights_by_context={"*|bear|3600s": {"momentum": 1.0}},
            regime="bear",
            horizon=3600,
            scores=SCORES,
        )
        assert weights == {
            "naive": 0.25,
            "momentum": 0.25,
            "ema-crossover": 0.25,
            "mean-reversion": 0.25,
        }

    def test_no_moe_hard_max(self):
        weights = ablation_weights(
            ABLATION_NO_MOE, weights_by_context=WEIGHTS, regime="bear", horizon=3600, scores=SCORES
        )
        assert weights == {"momentum": 1.0}  # reward_adjusted 0.50 > 0.30

    def test_no_moe_tie_broken_alphabetically(self):
        scores = (
            _score("momentum", "bear", 0.30),
            _score("ema-crossover", "bear", 0.30),
        )
        weights = ablation_weights(
            ABLATION_NO_MOE, weights_by_context=WEIGHTS, regime="bear", horizon=3600, scores=scores
        )
        assert weights == {"momentum": 1.0}  # empate -> el que cierra por orden alfabético

    def test_full_uses_regime_context(self):
        weights = ablation_weights(
            ABLATION_FULL, weights_by_context=WEIGHTS, regime="bear", horizon=3600, scores=SCORES
        )
        assert weights == {"momentum": 0.7, "naive": 0.3}

    def test_full_falls_back_to_global(self):
        weights = ablation_weights(
            ABLATION_FULL,
            weights_by_context={"*|-|3600s": {"momentum": 1.0}},
            regime="bear",
            horizon=3600,
            scores=SCORES,
        )
        assert weights == {"momentum": 1.0}

    def test_full_uniform_when_no_context(self):
        weights = ablation_weights(
            ABLATION_FULL, weights_by_context={}, regime="bear", horizon=3600, scores=SCORES
        )
        assert weights == {
            "naive": 0.25,
            "momentum": 0.25,
            "ema-crossover": 0.25,
            "mean-reversion": 0.25,
        }

    def test_none_when_regime_scoped_empty(self):
        scores = (_score("naive", "neutral", 0.05),)
        assert (
            ablation_weights(
                ABLATION_FULL, weights_by_context=WEIGHTS, regime="bear", horizon=3600, scores=scores
            )
            is None
        )


class TestPortfolio:
    COST = 0.0012

    def test_directional_pnl_and_cost(self):
        # ts1: mercado +1% (momentum acierta +, naive falla -)
        # ts2: mercado -0.2% (momentum falla -, naive acierta -)
        per_ts = [
            (1.0, {"momentum": (True, 0.01), "naive": (False, 0.01)}),
            (2.0, {"momentum": (False, -0.002), "naive": (True, -0.002)}),
        ]
        returns = portfolio_net_returns({"momentum": 0.7, "naive": 0.3}, per_ts, self.COST)
        assert returns == pytest.approx(
            [0.7 * 0.01 + 0.3 * -0.01 - self.COST, 0.7 * -0.002 + 0.3 * 0.002 - self.COST]
        )

    def test_skips_timestamps_with_missing_expert(self):
        per_ts = [
            (1.0, {"momentum": (True, 0.01), "naive": (False, 0.01)}),
            (2.0, {"momentum": (True, 0.02)}),  # naive faltante -> excluido
        ]
        returns = portfolio_net_returns({"momentum": 0.7, "naive": 0.3}, per_ts, self.COST)
        assert len(returns) == 1


class TestStats:
    def test_sharpe_known_series(self):
        assert sharpe_of([1.0, -1.0, 1.0, -1.0]) == pytest.approx(0.0)
        assert sharpe_of([1.0]) == 0.0
        assert sharpe_of([0.1, 0.1, 0.1]) == 0.0  # sin varianza

    def test_max_drawdown(self):
        # Serie acumulada: sube a 0.05, cae a 0.02, sube a 0.04, cae a 0.01.
        assert max_drawdown([0.05, -0.03, 0.02, -0.03]) == pytest.approx(-0.04)
        assert max_drawdown([0.1, 0.1, 0.1]) == 0.0

    def test_window_stats_math(self):
        # mercado +1% siempre; momentum acierta (70%), naive falla (30%).
        per_ts = [
            (i, {"momentum": (True, 0.01), "naive": (False, 0.01)}) for i in range(4)
        ]
        stats = ablation_window_stats(
            symbol="TEST",
            index=0,
            regime="bear",
            ablation=ABLATION_FULL,
            cost_bps=12,
            weights={"momentum": 0.7, "naive": 0.3},
            expected={"momentum": 0.0015, "naive": 0.0005},
            accuracy={"momentum": 0.9, "naive": 0.4},
            per_timestamp=per_ts,
        )
        assert stats.n == 4
        assert stats.gross_edge == pytest.approx(0.7 * 0.0015 + 0.3 * 0.0005)
        assert stats.accuracy == pytest.approx(0.7 * 0.9 + 0.3 * 0.4)
        assert stats.realized_gross == pytest.approx(0.7 * 0.01 + 0.3 * -0.01)
        assert stats.realized_net == pytest.approx(stats.realized_gross - 0.0012)

    def test_aggregate_n_weighted(self):
        windows = [
            _window(n=2, acc=0.6, gross=0.01, net=0.0),
            _window(n=8, acc=0.5, gross=-0.01, net=-0.002),
        ]
        stats = aggregate_ablation(windows, pooled_returns=[0.0, -0.002])
        assert stats.n == 10
        assert stats.accuracy == pytest.approx((0.6 * 2 + 0.5 * 8) / 10)
        assert stats.gross_edge == pytest.approx((0.01 * 2 + -0.01 * 8) / 10)
        assert stats.net_edge == pytest.approx((0.0 * 2 + -0.002 * 8) / 10)


def _window(*, n, gross, net, acc=0.5, ablation=ABLATION_NAIVE):
    return AblationWindow(
        symbol="TEST",
        index=0,
        regime="bear",
        ablation=ablation,
        cost_bps=12,
        n=n,
        accuracy=acc,
        gross_edge=gross,
        realized_gross=net + 0.0012,
        realized_net=net,
        sharpe=0.0,
        max_drawdown=0.0,
    )


class TestRender:
    def test_matrix_renders_rows_and_regimes(self):
        stats_by_symbol = {
            "TEST": {
                ABLATION_NAIVE: AblationStats(
                    ABLATION_NAIVE, 10, 0.51, 0.001, -0.0005, -0.2, -0.02
                ),
                ABLATION_FULL: AblationStats(
                    ABLATION_FULL, 10, 0.55, 0.002, -0.0001, 0.3, -0.01
                ),
            }
        }
        by_regime = {
            "TEST": {
                (ABLATION_NAIVE, "bear"): AblationStats(
                    ABLATION_NAIVE, 6, 0.51, 0.001, -0.0005, -0.2, -0.02
                ),
                (ABLATION_FULL, "bear"): AblationStats(
                    ABLATION_FULL, 6, 0.55, 0.002, -0.0001, 0.3, -0.01
                ),
            }
        }
        text = render_ablation_matrix(
            stats_by_symbol,
            by_regime,
            cost_bps={"TEST": 12},
            window_counts={"TEST": 8},
        )
        assert "ABLATION MATRIX" in text
        assert "== TEST (12 bps, 8 ventanas) ==" in text
        assert ABLATION_NAIVE in text
        assert ABLATION_FULL in text
        assert "por régimen:" in text
        assert "bear" in text
        assert "51.0%" in text  # acc 0.51
        assert "+0.10%" in text  # gross 0.001
        assert "-0.05%" in text  # net -0.0005
