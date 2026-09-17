"""Unit tests for PortfolioRiskManager: High-Water Mark daily circuit breaker, correlation guardrails, and macro velocity filtering."""

import pytest
from iot_machine_learning.infrastructure.adapters.market.zephyr.risk.portfolio_risk_manager import (
    PortfolioRiskConfig,
    PortfolioRiskManager,
)


def test_portfolio_profit_lock_not_triggered_below_activation():
    config = PortfolioRiskConfig(profit_lock_trigger_usd=10.0, max_giveback_pct=0.25)
    mgr = PortfolioRiskManager(initial_equity=100000.0, config=config)

    # Small gain of $5 (below $10 trigger)
    tripped, reason = mgr.update_equity(100005.0)
    assert not tripped
    assert not mgr.is_profit_locked

    # Drops back to $100001
    tripped, reason = mgr.update_equity(100001.0)
    assert not tripped
    assert not mgr.circuit_breaker_tripped


def test_portfolio_circuit_breaker_trips_after_peak_drawdown():
    config = PortfolioRiskConfig(profit_lock_trigger_usd=10.0, max_giveback_pct=0.25)
    mgr = PortfolioRiskManager(initial_equity=100000.0, config=config)

    # Reaches peak profit of $13.00 USD
    tripped, _ = mgr.update_equity(100013.0)
    assert not tripped
    assert mgr.is_profit_locked
    assert mgr.peak_equity == 100013.0

    # Max giveback = 13.00 * 0.25 = 3.25. Floor equity = 100013 - 3.25 = 100009.75.
    # Minor pull back to $100011.00 (giveback $2.00 <= $3.25): should NOT trip
    tripped, _ = mgr.update_equity(100011.0)
    assert not tripped
    assert not mgr.circuit_breaker_tripped

    # Equity pulls back to $100009.00 (below floor $100009.75): MUST TRIP!
    tripped, reason = mgr.update_equity(100009.0)
    assert tripped
    assert mgr.circuit_breaker_tripped
    assert "Portfolio Circuit Breaker Triggered" in reason
    assert "Preserved profit: $9.00" in reason

    # Stays tripped on subsequent ticks
    tripped_again, _ = mgr.update_equity(100008.0)
    assert tripped_again


def test_correlation_guardrail_blocks_same_side_in_cluster():
    config = PortfolioRiskConfig(max_cluster_positions=1)
    mgr = PortfolioRiskManager(initial_equity=100000.0, config=config)

    # Active short position in SPY
    open_positions = {"SPY": -1.0, "AAPL": 0.0}

    # Attempting to open another SHORT in QQQ (same cluster) -> VETO
    allowed, reason = mgr.check_correlation_guardrail("QQQ", "sell", open_positions)
    assert not allowed
    assert "Correlation Guardrail VETO" in reason
    assert "SPY(-1.0)" in reason

    # Attempting to open a LONG in QQQ -> ALLOWED
    allowed, reason = mgr.check_correlation_guardrail("QQQ", "buy", open_positions)
    assert allowed
    assert reason == ""

    # Asset outside cluster -> ALLOWED
    allowed, reason = mgr.check_correlation_guardrail("BTCUSDT", "sell", open_positions)
    assert allowed


def test_macro_velocity_filter_prevents_shorting_upward_rally():
    config = PortfolioRiskConfig(enable_macro_velocity_filter=True, macro_velocity_epsilon=0.0001)
    mgr = PortfolioRiskManager(initial_equity=100000.0, config=config)

    # Upward velocity (market bouncing)
    positive_velocity = 0.0015
    allowed_sell, reason_sell = mgr.check_macro_velocity("SPY", "sell", positive_velocity)
    assert not allowed_sell
    assert "Cannot open SHORT while macro velocity is positive" in reason_sell

    allowed_buy, _ = mgr.check_macro_velocity("SPY", "buy", positive_velocity)
    assert allowed_buy

    # Downward velocity (market falling)
    negative_velocity = -0.0015
    allowed_buy_down, reason_buy_down = mgr.check_macro_velocity("SPY", "buy", negative_velocity)
    assert not allowed_buy_down
    assert "Cannot open LONG while macro velocity is negative" in reason_buy_down

    allowed_sell_down, _ = mgr.check_macro_velocity("SPY", "sell", negative_velocity)
    assert allowed_sell_down


def test_can_execute_integration_with_portfolio_risk_manager():
    from types import SimpleNamespace
    from iot_machine_learning.infrastructure.adapters.market.zephyr.config import LiveBotConfig
    from iot_machine_learning.infrastructure.adapters.market.zephyr.execution import can_execute
    from iot_machine_learning.infrastructure.adapters.market.zephyr.models import LiveBotState

    cfg = LiveBotConfig(symbol="SPY")
    plan_sell = SimpleNamespace(action="EXECUTE", chosen_trajectory=SimpleNamespace(side="sell"))
    plan_buy = SimpleNamespace(action="EXECUTE", chosen_trajectory=SimpleNamespace(side="buy"))

    state_ok = LiveBotState(last_phi_moe=0.8, last_lambda_t=0.1, current_position=0.0)
    risk_mgr = PortfolioRiskManager(initial_equity=100000.0)

    # 1. Circuit breaker tripped in state -> VETO
    state_tripped = LiveBotState(last_phi_moe=0.8, last_lambda_t=0.1, current_position=0.0, portfolio_circuit_breaker_tripped=True)
    assert can_execute(plan_sell, cfg, state_tripped, 758.0, symbol="SPY", risk_mgr=risk_mgr) is False

    # 2. Positive macro velocity on sell order -> VETO
    assert can_execute(plan_sell, cfg, state_ok, 758.0, symbol="SPY", risk_mgr=risk_mgr, macro_velocity=0.002) is False
    # But buy order with positive velocity -> ALLOWED
    assert can_execute(plan_buy, cfg, state_ok, 758.0, symbol="SPY", risk_mgr=risk_mgr, macro_velocity=0.002) is True

    # 3. Correlation guardrail: SPY already short -> QQQ short is VETOED
    state_short_spy = LiveBotState(last_phi_moe=0.8, last_lambda_t=0.1, positions={"SPY": -1.0})
    assert can_execute(plan_sell, cfg, state_short_spy, 500.0, symbol="QQQ", risk_mgr=risk_mgr) is False
    # QQQ long is ALLOWED
    assert can_execute(plan_buy, cfg, state_short_spy, 500.0, symbol="QQQ", risk_mgr=risk_mgr) is True

