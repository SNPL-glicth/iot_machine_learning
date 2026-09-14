"""Unit tests for Forensic Fixes — Phase 4 (Medium Severity Bugs).

Fix 1: Multi-asset position isolation in LiveBotState (set_position/get_position/current_position).
Fix 2: Honest order cancellation reporting and fill-confirmed position closure logs.
Fix 3: Failsafe account blocking when broker security keys are missing or corrupt.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iot_machine_learning.infrastructure.adapters.market.alpaca.account import AlpacaAccount
from iot_machine_learning.infrastructure.adapters.market.alpaca.account_sync import (
    fetch_account_and_positions,
)
from iot_machine_learning.infrastructure.adapters.market.alpaca.order_client import (
    AlpacaOrderClient,
)
from iot_machine_learning.infrastructure.adapters.market.alpaca.order_models import OrderResponse
from iot_machine_learning.infrastructure.adapters.market.live_config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.live_runner_execution import (
    can_execute,
    perform_shutdown,
)
from iot_machine_learning.infrastructure.adapters.market.live_runner_models import LiveBotState
from iot_machine_learning.infrastructure.adapters.market.rosa_roja_market_handler import (
    RosaRojaMarketExecutionHandler,
)


# ==============================================================================
# FIX 1: Multi-Asset Position Isolation & Safe Computed Property
# ==============================================================================

def test_fix1_multi_asset_position_isolation_and_no_overwriting():
    """Confirms that iterating symbols during health checks never overwrites other positions."""
    state = LiveBotState()
    symbols = ["SPY", "AAPL", "QQQ"]

    # Simular ciclo de health check secuencial
    state.set_position("SPY", 10.0)
    state.set_position("AAPL", 0.0)
    state.set_position("QQQ", 5.0)

    # Cada símbolo debe mantener su posición aislada e independiente
    assert state.get_position("SPY") == 10.0
    assert state.get_position("AAPL") == 0.0
    assert state.get_position("QQQ") == 5.0
    assert state.active_positions_count == 2

    # La propiedad calculada current_position debe referenciar SPY de forma segura
    assert state.current_position == 10.0

    # Actualizar un símbolo secundario a 0 no debe resetear SPY ni current_position
    state.set_position("QQQ", 0.0)
    assert state.get_position("SPY") == 10.0
    assert state.current_position == 10.0
    assert state.active_positions_count == 1


def test_fix1_current_position_property_backward_compatibility():
    """Verifica compatibilidad con llamadas legacy a LiveBotState(current_position=...) y setters."""
    # Instanciación legacy con escalar
    s1 = LiveBotState(current_position=7.5)
    assert s1.get_position("SPY") == 7.5
    assert s1.current_position == 7.5

    # Asignación legacy vía setter
    s1.current_position = 0.0
    assert s1.get_position("SPY") == 0.0
    assert s1.current_position == 0.0

    # Estado con único activo diferente a SPY
    s2 = LiveBotState()
    s2.set_position("BTCUSDT", 0.05)
    assert s2.current_position == 0.05


def test_fix1_can_execute_and_shutdown_multi_asset_safety():
    """can_execute y perform_shutdown evalúan la canasta multiactivo sin cegueras por escalar."""
    cfg = LiveBotConfig(symbol="SPY", max_concurrent_positions=2)
    state = LiveBotState(last_phi_moe=0.8, last_lambda_t=0.01)
    plan = MagicMock(action="EXECUTE")

    # Sin símbolo especificado: si hay alguna posición abierta, bloquea
    state.set_position("AAPL", 5.0)
    assert can_execute(plan, cfg, state, current_price=100.0, symbol=None) is False

    # Con símbolo especificado que está plano: si hay cupo, permite
    assert can_execute(plan, cfg, state, current_price=100.0, symbol="SPY") is True

    # Si se alcanza max_concurrent_positions, bloquea
    state.set_position("SPY", 10.0)
    assert can_execute(plan, cfg, state, current_price=100.0, symbol="QQQ") is False


@pytest.mark.asyncio
async def test_fix1_perform_shutdown_flattens_all_open_positions():
    """perform_shutdown cierra todas las posiciones activas en positions dict, no solo SPY."""
    state = LiveBotState()
    state.set_position("SPY", 10.0)
    state.set_position("QQQ", 5.0)
    state.set_position("AAPL", 0.0)

    mock_client = AsyncMock()
    mock_client.close_position = AsyncMock(return_value={"status": "filled"})
    mock_handler = AsyncMock()
    cfg = LiveBotConfig(symbol="SPY")

    await perform_shutdown(
        handler=mock_handler,
        order_client=mock_client,
        feed=None,
        account=None,
        state=state,
        config=cfg,
        state_path=None,
    )

    # Debe haber llamado close_position para SPY y para QQQ, pero no para AAPL (qty=0)
    closed_symbols = [call.args[0] for call in mock_client.close_position.call_args_list]
    assert "SPY" in closed_symbols
    assert "QQQ" in closed_symbols
    assert "AAPL" not in closed_symbols


# ==============================================================================
# FIX 2: Honest Cancellation Accounting & Fill Confirmation
# ==============================================================================

@pytest.mark.asyncio
async def test_fix2_cancel_all_orders_reports_actual_success_without_silent_pass():
    """cancel_all_orders registra fallos y retorna el conteo real de canceladas, no len(matched)."""
    client = AlpacaOrderClient(api_key="key", api_secret="sec")

    o1 = MagicMock(spec=OrderResponse, id="ord-1", symbol="SPY")
    o2 = MagicMock(spec=OrderResponse, id="ord-2", symbol="SPY")
    o3 = MagicMock(spec=OrderResponse, id="ord-3", symbol="SPY")

    client.get_orders = AsyncMock(return_value=[o1, o2, o3])

    async def mock_cancel(order_id: str):
        if order_id == "ord-2":
            raise RuntimeError("HTTP 500: Alpaca order cancellation timeout")
        return MagicMock(id=order_id, status="canceled")

    client.cancel_order = AsyncMock(side_effect=mock_cancel)

    with patch("iot_machine_learning.infrastructure.adapters.market.alpaca.order_client.logger") as mock_log:
        cancelled_count = await client.cancel_all_orders(symbol="SPY")

        # De 3 órdenes, 1 falló: debe reportar exactamente 2 canceladas
        assert cancelled_count == 2
        # Verifica que la falla no fue silenciada con pass, sino registrada
        assert mock_log.warning.called
        assert mock_log.error.called


@pytest.mark.asyncio
async def test_fix2_handler_logs_closed_only_on_confirmed_fill(caplog):
    """trigger_emergency_flush no emite 'closed position' si la API del broker rechaza el cierre."""
    import logging
    caplog.set_level(logging.INFO)

    broker = AsyncMock()
    broker.get_position = AsyncMock(return_value={"symbol": "SPY", "qty": "10", "avg_entry_price": "500.0", "side": "long"})
    broker.cancel_all_orders = AsyncMock(return_value=1)

    handler = RosaRojaMarketExecutionHandler(
        broker_client=broker,
        account_equity=100000.0,
        symbol="SPY",
    )
    handler.get_reference_price_callback = lambda sym: 502.0

    # Caso A: Cierre rechazado por broker (status="rejected")
    broker.close_position = AsyncMock(return_value={"status": "rejected", "error": "Order rejected by exchange"})
    await handler.trigger_emergency_flush("test_rejection", symbol="SPY")

    # El log de confirmación NO debe haberse emitido
    assert not any("closed position of 10.0 for SPY confirmed by broker" in r.message for r in caplog.records)
    # Debe haber registrado ERROR por falta de confirmación
    assert any("broker did not confirm position closure for SPY" in r.message for r in caplog.records)

    caplog.clear()

    # Caso B: Cierre confirmado por broker (status="filled")
    broker.close_position = AsyncMock(return_value={"id": "fill_1", "status": "filled", "symbol": "SPY"})
    await handler.trigger_emergency_flush("test_fill", symbol="SPY")

    # Ahora sí debe emitir el log de confirmación
    assert any("closed position of 10.0 for SPY confirmed by broker" in r.message for r in caplog.records)


# ==============================================================================
# FIX 3: Failsafe Account Block Handling on Missing or Corrupt Keys
# ==============================================================================

@pytest.mark.asyncio
async def test_fix3_account_sync_failsafe_on_missing_keys():
    """Si la respuesta de Alpaca no incluye trading_blocked o account_blocked, asume bloqueado."""
    client = AsyncMock()
    # Payload incompleto/corrupto sin trading_blocked ni account_blocked
    client.get_account = AsyncMock(return_value={"equity": "10000.0", "cash": "5000.0"})
    client.get_positions = AsyncMock(return_value=[])

    fields, positions = await fetch_account_and_positions(client)

    # La política fail-safe debe marcar la cuenta como bloqueada y sospechosa
    assert fields["trading_blocked"] is True
    assert fields["account_blocked"] is True
    assert fields["suspicious_payload"] is True


@pytest.mark.asyncio
async def test_fix3_incomplete_payload_vetos_trading_in_can_execute():
    """Un estado con account_blocked=True debido a payload sospechoso bloquea can_execute."""
    client = AsyncMock()
    client.get_account = AsyncMock(return_value={"equity": "10000.0"})  # Incompleto
    client.get_positions = AsyncMock(return_value=[])

    account = AlpacaAccount(client=client, auto_sync_interval=0)
    # Sincronizar cuenta
    await account.sync_now()

    # is_tradeable debe retornar False
    tradeable = await account.is_tradeable()
    assert tradeable is False

    # El runner bloqueará entradas
    state = LiveBotState(last_phi_moe=0.9, last_lambda_t=0.01)
    state.account_blocked = not tradeable

    cfg = LiveBotConfig(symbol="SPY")
    plan = MagicMock(action="EXECUTE")

    # can_execute debe vetar inmediatamente la orden
    assert can_execute(plan, cfg, state, current_price=500.0, symbol="SPY") is False
