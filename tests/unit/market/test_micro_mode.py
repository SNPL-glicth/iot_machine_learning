"""Unit tests for Micro-Mode (Modo Carrito) low-capital execution and standard scaling."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from iot_machine_learning.infrastructure.adapters.market.zephyr.engines.rosa_roja.market_handler import RosaRojaMarketExecutionHandler
from iot_machine_learning.infrastructure.adapters.market.zephyr.execution import can_execute
from iot_machine_learning.infrastructure.adapters.market.zephyr.models import LiveBotState
from iot_machine_learning.infrastructure.adapters.market.zephyr.config import LiveBotConfig


@pytest.mark.asyncio
async def test_micro_mode_fractional_dispatch_and_scaling():
    mock_broker = AsyncMock()
    mock_broker.submit_order = AsyncMock(return_value={"id": "test_order_1"})

    # 1. Micro mode ($20 USD equity)
    handler = RosaRojaMarketExecutionHandler(
        broker_client=mock_broker,
        account_equity=20.0,
        symbol="SPY",
        max_position_pct=0.25,
    )
    handler.get_reference_price_callback = lambda sym: 760.0

    plan_buy = MagicMock()
    plan_buy.action = "EXECUTE"
    plan_buy.regime_alert = False
    plan_buy.symbol = "SPY"
    plan_buy.global_confidence = 0.95
    plan_buy.invalidation_step = 10
    plan_buy.envelope = MagicMock(magnitude=0.25, bounds={"stop_pct": 0.0035, "target_pct": 0.0035}, max_steps=10)
    plan_buy.chosen_trajectory = MagicMock(side="buy", length=5, terminal_state=MagicMock(step_index=1, state_vector=[0.003]))

    # Buy order in micro mode -> must send notional=5.0 and qty=0.0 without bracket
    res_buy = await handler.dispatch_execution(plan_buy, symbol="SPY")
    assert res_buy is True
    call_kwargs = mock_broker.submit_order.call_args.kwargs
    assert call_kwargs["notional"] == 5.0
    assert call_kwargs["qty"] == 0.0
    assert call_kwargs["order_class"] is None

    # Sell (Short) order in micro mode -> must be vetoed
    plan_sell = MagicMock()
    plan_sell.action = "EXECUTE"
    plan_sell.regime_alert = False
    plan_sell.symbol = "SPY"
    plan_sell.global_confidence = 0.95
    plan_sell.invalidation_step = 10
    plan_sell.envelope = MagicMock(magnitude=0.25, bounds={"stop_pct": 0.0035, "target_pct": 0.0035}, max_steps=10)
    plan_sell.chosen_trajectory = MagicMock(side="sell", length=5, terminal_state=MagicMock(step_index=2, state_vector=[-0.003]))

    res_sell = await handler.dispatch_execution(plan_sell, symbol="SPY")
    assert res_sell is False

    # 2. Scaled to standard mode ($2000 USD equity)
    handler.update_equity(2000.0)
    res_std = await handler.dispatch_execution(plan_buy, symbol="SPY")
    assert res_std is True
    call_kwargs_std = mock_broker.submit_order.call_args.kwargs
    assert call_kwargs_std["order_class"] == "bracket"
    assert "notional" not in call_kwargs_std or call_kwargs_std.get("notional") is None
    assert call_kwargs_std["qty"] >= 1.0


def test_can_execute_micro_mode_guardrail():
    config = LiveBotConfig(symbol="SPY")
    state = LiveBotState(equity=20.0, positions={"SPY": 0.0}, last_phi_moe=0.95)

    plan_sell = MagicMock()
    plan_sell.action = "EXECUTE"
    plan_sell.side = "sell"

    # Micro mode vetoes short opening
    allowed = can_execute(plan_sell, config, state, current_price=760.0, symbol="SPY")
    assert allowed is False

    # Standard mode allows short opening
    state.equity = 2500.0
    allowed_std = can_execute(plan_sell, config, state, current_price=760.0, symbol="SPY")
    assert allowed_std is True
