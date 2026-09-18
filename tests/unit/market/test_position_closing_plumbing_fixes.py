"""Unit tests for Position Closing Plumbing Fixes (Fix 1, Fix 2, Fix 3)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from iot_machine_learning.infrastructure.adapters.market.zephyr.config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.zephyr.execution import can_execute
from iot_machine_learning.infrastructure.adapters.market.zephyr.models import LiveBotState
from iot_machine_learning.infrastructure.adapters.market.zephyr.engines.rosa_roja_execution.market_handler import RosaRojaMarketExecutionHandler
from iot_machine_learning.infrastructure.adapters.market.alpaca.order_client_market import AlpacaMarketMixin


@pytest.mark.asyncio
async def test_fix1_same_tick_reentry_lock_and_cooldown():
    """Fix 1: Verifies is_closing prevents re-entry in the same tick and during 45s cooldown."""
    state = LiveBotState()
    state.last_phi_moe = 0.8  # Satisfies config.phi_moe_threshold
    config = LiveBotConfig(symbol="SPY")
    plan = MagicMock(action="EXECUTE", side="buy", chosen_trajectory=None)

    # Initial state: can execute
    assert can_execute(plan, config, state, current_price=500.0, symbol="SPY") is True

    # 1. Mark closing initiated
    state.mark_closing("SPY")
    assert state.is_closing["SPY"] is True
    # Immediate check in the same tick: must be vetoed
    assert can_execute(plan, config, state, current_price=500.0, symbol="SPY") is False

    # 2. Mark close confirmed by broker
    state.mark_close_confirmed("SPY")
    # Still within cooldown (e.g. 1 second elapsed < 45s): must be vetoed
    assert state.is_symbol_closing("SPY", cooldown_sec=45.0) is True
    assert can_execute(plan, config, state, current_price=500.0, symbol="SPY") is False

    # 3. Simulate cooldown expired (46s later)
    state.close_confirmed_at["SPY"] = time.time() - 46.0
    assert state.is_symbol_closing("SPY", cooldown_sec=45.0) is False
    assert can_execute(plan, config, state, current_price=500.0, symbol="SPY") is True

    # 4. Error survival test: if closing started but never confirmed (e.g. crash/exception)
    state.mark_closing("AAPL")
    # No mark_close_confirmed called!
    assert state.is_symbol_closing("AAPL", cooldown_sec=45.0) is True
    # Even after 100 seconds, it remains locked because it was never confirmed
    assert can_execute(plan, config, state, current_price=150.0, symbol="AAPL") is False


@pytest.mark.asyncio
async def test_fix2_close_position_sends_cancel_orders_true():
    """Fix 2: Verifies close_position includes params={'cancel_orders': 'true'}."""
    client = AlpacaMarketMixin(api_key="test_key", api_secret="test_secret")
    client._request = AsyncMock(return_value={"symbol": "SPY", "status": "closed"})

    res = await client.close_position("SPY")
    assert res["status"] == "closed"

    client._request.assert_awaited_once_with(
        "DELETE",
        "/v2/positions/SPY",
        params={"cancel_orders": "true"},
        weight=1,
    )


@pytest.mark.asyncio
async def test_fix3_emergency_flush_resilient_to_cancel_all_orders_failure():
    """Fix 3: Verifies emergency flush proceeds to close_position even if cancel_all_orders fails."""
    broker = MagicMock()
    broker.cancel_order = AsyncMock()
    # Simula falla HTTP 503 / 429 de Alpaca en cancel_all_orders
    broker.cancel_all_orders = AsyncMock(side_effect=RuntimeError("HTTP 503: Service Unavailable"))
    broker.get_position = AsyncMock(return_value={"qty": "10.0"})
    broker.close_position = AsyncMock()

    state = LiveBotState()
    handler = RosaRojaMarketExecutionHandler(
        broker_client=broker,
        account_equity=100000.0,
        symbol="SPY",
        state=state,
    )

    # Trigger emergency flush
    await handler.trigger_emergency_flush(reason="test_http_failure", symbol="SPY")

    # 1. cancel_all_orders was attempted
    broker.cancel_all_orders.assert_awaited_once_with(symbol="SPY")

    # 2. Execution continued and close_position was STILL awaited!
    broker.get_position.assert_awaited_once_with("SPY")
    broker.close_position.assert_awaited_once_with("SPY")

    # 3. Post-close confirmation was registered
    assert "SPY" in state.close_confirmed_at
    assert state.is_symbol_closing("SPY", cooldown_sec=45.0) is True
