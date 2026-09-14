"""Regression tests for Short PnL Unification, Trade 4 AAPL Stop-Loss, and Calibrated Profit Lock."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from iot_machine_learning.infrastructure.adapters.market.portfolio_risk_manager import (
    PortfolioRiskConfig,
    PortfolioRiskManager,
)
from iot_machine_learning.infrastructure.adapters.market.rosa_roja_market_handler import (
    RosaRojaMarketExecutionHandler,
    calculate_unrealized_pnl,
)


# ==============================================================================
# TEST 1: Unified Unrealized PnL Mathematics (Short vs Long)
# ==============================================================================

def test_calculate_unrealized_pnl_long_and_short():
    """Confirms unified formula calculates correct sign for Long and Short across conventions."""
    # 1. Long position: entry 333.71, price rises to 334.79 (+1.08/sh * 30 = +$32.40)
    pnl_long_win = calculate_unrealized_pnl("long", 30, 333.71, 334.79)
    assert round(pnl_long_win, 2) == 32.40

    # Long position: entry 333.71, price falls to 332.71 (-1.00/sh * 30 = -$30.00)
    pnl_long_loss = calculate_unrealized_pnl("long", 30, 333.71, 332.71)
    assert round(pnl_long_loss, 2) == -30.00

    # 2. Short position (Alpaca convention: qty > 0, side="short")
    # Entry 333.71, price rises to 334.79 -> Real loss of -$32.40 (Trade 4 AAPL scenario)
    pnl_short_loss = calculate_unrealized_pnl("short", 30, 333.71, 334.79)
    assert round(pnl_short_loss, 2) == -32.40

    # Short position: entry 333.71, price falls to 332.71 -> Real gain of +$30.00
    pnl_short_win = calculate_unrealized_pnl("short", 30, 333.71, 332.71)
    assert round(pnl_short_win, 2) == 30.00

    # 3. Short position (Negative qty convention: qty = -30, side="")
    pnl_neg_qty_loss = calculate_unrealized_pnl("", -30, 333.71, 334.79)
    assert round(pnl_neg_qty_loss, 2) == -32.40

    pnl_neg_qty_win = calculate_unrealized_pnl("", -30, 333.71, 332.71)
    assert round(pnl_neg_qty_win, 2) == 30.00

    # 4. Side="sell" with positive qty
    pnl_sell_side = calculate_unrealized_pnl("sell", 30, 333.71, 334.79)
    assert round(pnl_sell_side, 2) == -32.40

    # 5. Edge cases / Invalid data
    assert calculate_unrealized_pnl("short", 0, 333.71, 334.79) == 0.0
    assert calculate_unrealized_pnl("short", 30, 0, 334.79) == 0.0
    assert calculate_unrealized_pnl("short", 30, 333.71, 0) == 0.0


# ==============================================================================
# TEST 2: Exact Reproduction of Trade 4 AAPL and $10.00 Stop-Loss Interception
# ==============================================================================

@pytest.mark.asyncio
async def test_trade4_aapl_software_stop_loss_intercepts_at_10_dollars():
    """Reproduces Trade 4 AAPL: Short 30 @ $333.71.

    Confirms:
    1. Loss is calculated as negative (not positive).
    2. Software Stop-Loss triggers when loss crosses -$10.00 ($334.05).
    3. Position is closed before ever reaching $334.79 (-$32.40).
    """
    broker = MagicMock()
    # Mock Alpaca get_position response for AAPL short
    broker.get_position = AsyncMock(return_value={
        "symbol": "AAPL",
        "qty": "30",
        "side": "short",
        "avg_entry_price": "333.71",
    })
    broker.cancel_all_orders = AsyncMock(return_value=2)
    broker.close_position = AsyncMock(return_value={"status": "filled", "symbol": "AAPL"})

    handler = RosaRojaMarketExecutionHandler(
        broker_client=broker,
        account_equity=100000.0,
        symbol="AAPL",
        max_stop_loss_usd=10.00,
    )

    # Step 1: At entry price $333.71, loss is $0.00 -> not closed
    closed_at_entry = await handler.check_trailing_profit(333.71, symbol="AAPL")
    assert closed_at_entry is False
    broker.close_position.assert_not_called()

    # Step 2: Price rises slightly to $333.90 (-$5.70 loss) -> below $10 threshold -> not closed
    closed_at_333_90 = await handler.check_trailing_profit(333.90, symbol="AAPL")
    assert closed_at_333_90 is False
    broker.close_position.assert_not_called()

    # Step 3: Price rises to $334.05 -> Loss is (333.71 - 334.05) * 30 = -$10.20 <= -$10.00!
    # MUST TRIGGER SOFTWARE STOP-LOSS!
    closed_at_334_05 = await handler.check_trailing_profit(334.05, symbol="AAPL")
    assert closed_at_334_05 is True
    broker.cancel_all_orders.assert_called_once_with(symbol="AAPL")
    broker.close_position.assert_called_once_with("AAPL")

    # Step 4: Verify the runaway loss price $334.79 would have calculated -$32.40
    pnl_at_runaway = calculate_unrealized_pnl("short", 30, 333.71, 334.79)
    assert round(pnl_at_runaway, 2) == -32.40


# ==============================================================================
# TEST 3: Calibrated Portfolio Profit Lock Trigger at $12.00 USD
# ==============================================================================

def test_calibrated_portfolio_profit_lock_at_12_dollars():
    """Verifies that with trigger=$12.00, the circuit breaker arms on a +$23.65 peak

    and trips at +$17.74 (stopping giveback at 25%).
    """
    cfg = PortfolioRiskConfig(
        profit_lock_trigger_usd=12.0,   # Calibrated to $12.00
        max_giveback_pct=0.25,
        max_daily_loss_usd=50.0,
    )
    # Session baseline
    initial_eq = 100003.61
    mgr = PortfolioRiskManager(initial_equity=initial_eq, config=cfg)

    # 1. Trade 2 & 3: Equity reaches $100,020.11 (+16.50 profit) -> Arms profit lock!
    tripped, _ = mgr.update_equity(100020.11)
    assert tripped is False
    assert mgr.is_profit_locked is True

    # 2. Peak reached at 18:04:00 UTC: $100,027.26 (+23.65 profit)
    tripped, _ = mgr.update_equity(100027.26)
    assert tripped is False
    assert mgr.peak_equity == 100027.26

    # Maximum giveback allowed: 25% of $23.65 = $5.9125 -> floor = $100,021.35
    # Equity at $100,022.00 (within 25% tolerance) -> not tripped
    tripped, _ = mgr.update_equity(100022.00)
    assert tripped is False

    # 3. At 18:06:00 UTC: Equity fell to $100,020.10 <= floor $100,021.35
    # MUST TRIP CIRCUIT BREAKER!
    tripped, reason = mgr.update_equity(100020.10)
    assert tripped is True
    assert "Portfolio Circuit Breaker Triggered" in reason
    assert "Peak profit was $23.65" in reason
    assert "floor $100021.35" in reason
    assert mgr.circuit_breaker_tripped is True
