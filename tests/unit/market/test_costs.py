"""FASE 9.2 — costos reales y edge después de costos (módulo puro)."""

import pytest
from iot_machine_learning.domain.entities.market.costs import (
    COST_PROFILES,
    DEFAULT_STOCK_COSTS,
    EDGE_COST_NEGATIVE,
    EDGE_COST_POSITIVE,
    EDGE_GROSS_NEGATIVE,
    EDGE_RISK_ADJUSTED_POSITIVE,
    EDGE_RISK_NEGATIVE,
    CostModel,
    classify_edge,
    edge_ladder_index,
)


class TestCostModel:
    def test_total_bps_round_trip(self):
        model = CostModel(spread_bps=4.0, slippage_bps=5.0, commission_bps=3.0)
        assert model.total_bps == 12
        assert model.total() == pytest.approx(0.0012)

    def test_net(self):
        model = CostModel(spread_bps=4.0, slippage_bps=5.0, commission_bps=3.0)
        assert model.net(0.0012) == pytest.approx(0.0)

    def test_invalid_bps_rejected(self):
        with pytest.raises(ValueError):
            CostModel(spread_bps=-1.0)
        with pytest.raises(ValueError):
            CostModel(commission_bps=2000.0)

    def test_string_shows_breakdown(self):
        assert "12bps" in str(DEFAULT_STOCK_COSTS)
        assert "spread 4" in str(DEFAULT_STOCK_COSTS)


class TestCostProfiles:
    def test_stocks_share_profile(self):
        assert COST_PROFILES["NVDA"] == COST_PROFILES["AAPL"] == DEFAULT_STOCK_COSTS
        assert COST_PROFILES["AMD"].total_bps == 12

    def test_crypto_higher_cost(self):
        btc = COST_PROFILES["BTC-USD"]
        assert btc.total_bps == 24
        assert btc.commission_bps == 20.0

    def test_unknown_symbol_not_present(self):
        assert "UNKNOWN" not in COST_PROFILES


class TestClassifyEdge:
    def test_gross_negative_when_gross_loses(self):
        assert classify_edge(-0.001, -0.002) == EDGE_GROSS_NEGATIVE
        assert classify_edge(0.0, 0.001) == EDGE_GROSS_NEGATIVE

    def test_cost_negative_when_costs_kill_signal(self):
        # El caso brutal: señal positiva en bruto, muerta a los costos.
        assert classify_edge(0.0012, 0.0) == EDGE_COST_NEGATIVE
        assert classify_edge(0.0012, -0.0001) == EDGE_COST_NEGATIVE

    def test_cost_positive_without_risk_info(self):
        assert classify_edge(0.002, 0.0008, sharpe=None) == EDGE_COST_POSITIVE

    def test_risk_negative_when_inconsistent(self):
        assert classify_edge(0.002, 0.0008, sharpe=0.1, sharpe_threshold=0.5) == EDGE_RISK_NEGATIVE

    def test_risk_adjusted_positive(self):
        assert (
            classify_edge(0.002, 0.0008, sharpe=1.2, sharpe_threshold=0.5)
            == EDGE_RISK_ADJUSTED_POSITIVE
        )

    def test_threshold_exact_boundary(self):
        assert classify_edge(0.002, 0.0008, sharpe=0.5) == EDGE_RISK_ADJUSTED_POSITIVE
        assert classify_edge(0.002, 0.0008, sharpe=0.49) == EDGE_RISK_NEGATIVE

    def test_ladder_order(self):
        ladder = (
            EDGE_GROSS_NEGATIVE,
            EDGE_COST_NEGATIVE,
            EDGE_RISK_NEGATIVE,
            EDGE_COST_POSITIVE,
            EDGE_RISK_ADJUSTED_POSITIVE,
        )
        idx = [edge_ladder_index(e) for e in ladder]
        assert idx == sorted(idx)
