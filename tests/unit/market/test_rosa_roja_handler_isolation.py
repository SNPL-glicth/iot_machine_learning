from unittest.mock import AsyncMock, MagicMock

import pytest

from iot_machine_learning.infrastructure.adapters.market.zephyr.engines.rosa_roja_execution.market_handler import RosaRojaMarketExecutionHandler


@pytest.mark.asyncio
async def test_order_tracking_and_flush_isolation():
    broker = MagicMock()
    broker.cancel_order = AsyncMock()
    broker.cancel_all_orders = AsyncMock(return_value=1)
    broker.get_position = AsyncMock(return_value={"qty": "10.0"})
    broker.close_position = AsyncMock()

    handler = RosaRojaMarketExecutionHandler(
        broker_client=broker,
        account_equity=100000.0,
        symbol="SPY",
    )

    # Track orders for two different symbols
    order_spy = {"id": "ord_spy_1", "symbol": "SPY"}
    order_aapl = {"id": "ord_aapl_1", "symbol": "AAPL"}

    handler._track_order(order_spy, symbol="SPY")
    handler._track_order(order_aapl, symbol="AAPL")

    assert "SPY" in handler._active_orders
    assert "AAPL" in handler._active_orders
    assert "ord_spy_1" in handler._active_orders["SPY"]
    assert "ord_aapl_1" in handler._active_orders["AAPL"]

    # Trigger emergency flush strictly for SPY
    await handler.trigger_emergency_flush(reason="test_flush", symbol="SPY")

    # Verify that ord_spy_1 was cancelled, but ord_aapl_1 was NOT touched!
    broker.cancel_order.assert_awaited_once_with("ord_spy_1")
    assert "ord_spy_1" not in handler._active_orders.get("SPY", {})
    assert "ord_aapl_1" in handler._active_orders["AAPL"]

    # Verify broker.cancel_all_orders called strictly for SPY
    broker.cancel_all_orders.assert_awaited_once_with(symbol="SPY")
    broker.close_position.assert_awaited_once_with("SPY")
