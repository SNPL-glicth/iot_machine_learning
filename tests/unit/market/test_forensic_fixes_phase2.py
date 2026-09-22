"""Unit and integration tests for Forensic Fixes Phase 2 (Fix 1, Fix 2, Fix 3)."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets

from iot_machine_learning.infrastructure.adapters.market.zephyr.config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.zephyr.execution import can_execute
from iot_machine_learning.infrastructure.adapters.market.zephyr.models import LiveBotState
from iot_machine_learning.infrastructure.adapters.market.zephyr.engines.rosa_roja.market_handler import (
    PriceUnavailableError,
    RosaRojaMarketExecutionHandler,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.telemetry.server import (
    TelemetryBroadcaster,
    create_telemetry_server,
)
from domain.entities.rosa_roja.execution import (
    ActionEnvelope,
    ExecutionPlan,
)


# ============================================================================
# FIX 1: TUI Panic Button Callback & WebSocket Resiliency
# ============================================================================

@pytest.mark.asyncio
async def test_fix1_tui_panic_button_executes_flush_and_sends_success_response():
    """Fix 1: Verifies EMERGENCY_FLUSH command over WS executes flush and notifies TUI."""
    handler = MagicMock()
    handler.trigger_emergency_flush = AsyncMock()

    runner = MagicMock()
    runner._handler = handler
    runner._symbols = ["SPY", "QQQ"]
    runner.config = LiveBotConfig(symbol="SPY")

    # Start telemetry server on dynamic localhost port
    port = 8791
    server = await create_telemetry_server(runner, host="127.0.0.1", port=port)

    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            # Send emergency flush command
            await ws.send(json.dumps({"command": "EMERGENCY_FLUSH"}))

            # Receive response from telemetry server
            resp_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            resp = json.loads(resp_raw)

            assert resp.get("type") == "command_response"
            assert resp.get("command") == "EMERGENCY_FLUSH"
            assert resp.get("status") == "success"

            # Verify trigger_emergency_flush was called for both active symbols
            assert handler.trigger_emergency_flush.await_count == 2
            handler.trigger_emergency_flush.assert_any_await("TUI Emergency Flush", symbol="SPY")
            handler.trigger_emergency_flush.assert_any_await("TUI Emergency Flush", symbol="QQQ")
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_fix1_tui_panic_button_error_handling_does_not_drop_ws():
    """Fix 1: Verifies that if trigger_emergency_flush fails, error is reported and WS stays alive."""
    handler = MagicMock()
    handler.trigger_emergency_flush = AsyncMock(side_effect=RuntimeError("Alpaca 500 Server Error"))

    runner = MagicMock()
    runner._handler = handler
    runner._symbols = ["SPY"]
    runner.config = LiveBotConfig(symbol="SPY")

    port = 8792
    server = await create_telemetry_server(runner, host="127.0.0.1", port=port)

    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            # Send emergency flush that will encounter broker error
            await ws.send(json.dumps({"command": "EMERGENCY_FLUSH"}))

            resp_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            resp = json.loads(resp_raw)

            assert resp.get("type") == "command_response"
            assert resp.get("command") == "EMERGENCY_FLUSH"
            assert resp.get("status") == "error"
            assert "Alpaca 500 Server Error" in resp.get("error", "")

            # CRITICAL CHECK: WebSocket connection is STILL OPEN and responsive!
            # Sending a subsequent command succeeds
            await ws.send(json.dumps({"command": "PAUSE"}))
            resp_pause = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
            assert resp_pause.get("command") == "PAUSE"
            assert resp_pause.get("status") == "success"
    finally:
        await server.stop()


# ============================================================================
# FIX 2: Removal of Hardcoded $759.0 Reference Price Fallback
# ============================================================================

def test_fix2_reference_price_never_returns_759_and_raises_unavailable():
    """Fix 2: Verifies _get_reference_price raises PriceUnavailableError when no feed/cache."""
    handler = RosaRojaMarketExecutionHandler(
        broker_client=MagicMock(),
        account_equity=100000.0,
        symbol="BTC/USD",
    )
    # No callback installed
    handler.get_reference_price_callback = None

    with pytest.raises(PriceUnavailableError) as exc_info:
        handler._get_reference_price("BTC/USD")
    assert "Market reference price unavailable for symbol 'BTC/USD'" in str(exc_info.value)

    # Callback returns 0 or invalid price
    handler.get_reference_price_callback = lambda sym: 0.0
    with pytest.raises(PriceUnavailableError):
        handler._get_reference_price("BTC/USD")


def test_fix2_reference_price_uses_stale_cache_within_ttl():
    """Fix 2: Verifies valid price is cached and used as STALE within 30s TTL, expires after."""
    handler = RosaRojaMarketExecutionHandler(
        broker_client=MagicMock(),
        account_equity=100000.0,
        symbol="BTC/USD",
    )

    # 1. Successful callback fetches 65000.0
    handler.get_reference_price_callback = lambda sym: 65000.0
    p = handler._get_reference_price("BTC/USD")
    assert p == 65000.0
    assert "BTC/USD" in handler._last_known_prices

    # 2. Callback suddenly fails (returns None or 0.0)
    handler.get_reference_price_callback = lambda sym: 0.0

    # Within 30s (e.g. 5 seconds elapsed): STALE cache is accepted safely
    handler._last_known_prices["BTC/USD"] = (65000.0, time.time() - 5.0)
    stale_price = handler._get_reference_price("BTC/USD", max_stale_age_sec=30.0)
    assert stale_price == 65000.0

    # Expired cache (> 30s): must raise PriceUnavailableError
    handler._last_known_prices["BTC/USD"] = (65000.0, time.time() - 35.0)
    with pytest.raises(PriceUnavailableError):
        handler._get_reference_price("BTC/USD", max_stale_age_sec=30.0)


@pytest.mark.asyncio
async def test_fix2_order_dispatch_safely_aborts_on_price_unavailable():
    """Fix 2: Verifies order execution returns False and aborts when price is unavailable."""
    broker = MagicMock()
    broker.submit_order = AsyncMock()

    handler = RosaRojaMarketExecutionHandler(
        broker_client=broker,
        account_equity=100000.0,
        symbol="BTC/USD",
    )
    handler.get_reference_price_callback = None  # Price unavailable

    envelope = ActionEnvelope(
        magnitude=0.1,
        bounds={"stop_pct": 0.02, "target_pct": 0.04},
        max_steps=10,
        metadata={},
    )
    traj = MagicMock(terminal_state=MagicMock(state_vector=[66000.0]), length=5)
    plan = ExecutionPlan.EXECUTE(
        trajectory=traj,
        confidence=0.85,
        envelope=envelope,
        invalidation_step=10,
    )

    # Dispatch should return False without crashing and without submitting broker order
    dispatched = await handler.dispatch_execution(plan, symbol="BTC/USD")
    assert dispatched is False
    broker.submit_order.assert_not_called()


# ============================================================================
# FIX 3: Empty Cache Emergency Flush Inversion (err=3388.00 audit log scenario)
# ============================================================================

@pytest.mark.asyncio
async def test_fix3_empty_cache_queries_broker_and_executes_flush_on_losing_position():
    """Fix 3: Simulates forensic audit log err=3388.00 with empty cache and losing position.
    Verifies flush is EXECUTED, not suppressed!"""
    broker = MagicMock()
    # Cache is empty/None in handler!
    # Direct broker query returns active losing position
    broker.get_position = AsyncMock(return_value={
        "symbol": "SPY",
        "qty": "10.0",
        "side": "long",
        "avg_entry_price": "510.0",
    })
    broker.cancel_all_orders = AsyncMock(return_value=1)
    broker.close_position = AsyncMock(return_value={"status": "closed"})

    handler = RosaRojaMarketExecutionHandler(
        broker_client=broker,
        account_equity=100000.0,
        symbol="SPY",
    )
    # Reference price is 505.0 -> loss of (505 - 510) * 10 = -$50.00 (< -$3.00)
    handler.get_reference_price_callback = lambda sym: 505.0
    # Crucial: _cached_positions is None (just cleared after entry)
    handler._cached_positions["SPY"] = None

    # Plan with exact audit log deviation message
    plan = ExecutionPlan.EMERGENCY_FLUSH(
        reason="Reactive_Trajectory_Deviation: err=3388.00 > 5.00"
    )

    # In the old code, this returned True ("maintaining position") and ignored the error!
    # In the new code, it MUST proceed with flush and return False
    result = await handler.dispatch_execution(plan, symbol="SPY")
    assert result is False

    # Verify emergency liquidation was called on broker
    broker.get_position.assert_awaited()
    broker.close_position.assert_awaited_with("SPY")


@pytest.mark.asyncio
async def test_fix3_empty_cache_and_broker_failure_assumes_risk_and_flushes():
    """Fix 3: When cache is empty and broker get_position fails, default assumes risk and flushes."""
    broker = MagicMock()
    broker.get_position = AsyncMock(side_effect=RuntimeError("Broker Timeout"))
    broker.cancel_all_orders = AsyncMock(return_value=0)
    broker.close_position = AsyncMock()

    handler = RosaRojaMarketExecutionHandler(
        broker_client=broker,
        account_equity=100000.0,
        symbol="SPY",
    )
    handler._cached_positions["SPY"] = None

    plan = ExecutionPlan.EMERGENCY_FLUSH(
        reason="Reactive_Trajectory_Deviation: err=3388.00 > 5.00"
    )

    # Must proceed with flush rather than assuming a winning position!
    with patch.object(handler, "trigger_emergency_flush", new_callable=AsyncMock) as mock_flush:
        result = await handler.dispatch_execution(plan, symbol="SPY")
        assert result is False
        mock_flush.assert_awaited_once_with(
            reason="Reactive_Trajectory_Deviation: err=3388.00 > 5.00",
            symbol="SPY",
        )


@pytest.mark.asyncio
async def test_fix3_empty_cache_maintains_only_if_broker_confirms_winning():
    """Fix 3: Sovereign Master Equation ensures EMERGENCY_FLUSH triggers liquidation without PnL override."""
    broker = MagicMock()
    broker.get_position = AsyncMock(return_value={
        "symbol": "SPY",
        "qty": "10.0",
        "side": "long",
        "avg_entry_price": "500.0",
    })
    broker.cancel_order = AsyncMock(return_value=True)
    broker.cancel_all_orders = AsyncMock(return_value=1)
    broker.close_position = AsyncMock(return_value=True)

    handler = RosaRojaMarketExecutionHandler(
        broker_client=broker,
        account_equity=100000.0,
        symbol="SPY",
    )
    handler.get_reference_price_callback = lambda sym: 505.0
    handler._cached_positions["SPY"] = None

    plan = ExecutionPlan.EMERGENCY_FLUSH(
        reason="Reactive_Trajectory_Deviation: err=2.00 > 1.00"
    )

    result = await handler.dispatch_execution(plan, symbol="SPY")
    assert result is False
    broker.close_position.assert_called_once_with("SPY")


# ============================================================================
# INTERACTION TEST: Emergency Flush vs. Fix 1 Post-Close Cooldown
# ============================================================================

def test_emergency_flush_bypasses_post_close_cooldown():
    """Interaction: Confirms can_execute allows EMERGENCY_FLUSH through even if cooldown is active."""
    state = LiveBotState()
    state.mark_closing("SPY")
    state.mark_close_confirmed("SPY")  # In active 45s cooldown

    config = LiveBotConfig(symbol="SPY")

    # Regular EXECUTE plan is blocked by cooldown
    exec_plan = MagicMock(action="EXECUTE")
    assert can_execute(exec_plan, config, state, current_price=500.0, symbol="SPY") is False

    # EMERGENCY_FLUSH plan is NEVER blocked by cooldown
    flush_plan = ExecutionPlan.EMERGENCY_FLUSH(reason="Emergency")
    assert can_execute(flush_plan, config, state, current_price=500.0, symbol="SPY") is True
