"""Integration tests para LiveBotRunner con mocks."""

from __future__ import annotations

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from iot_machine_learning.infrastructure.adapters.market.live_config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.live_runner import LiveBotRunner, LiveBotState
from iot_machine_learning.core.orchestration.rosa_roja.engine import RosaRojaEngine
from iot_machine_learning.infrastructure.adapters.market.binance.ws_feed import BinanceWSFeed
from iot_machine_learning.infrastructure.adapters.market.binance.order_client import BinanceOrderClient
from iot_machine_learning.infrastructure.adapters.market.binance.account import BinanceAccount
from iot_machine_learning.domain.entities.market.observations import Quote, Trade, Candle
from iot_machine_learning.core.orchestration.rosa_roja.domain.execution import ExecutionPlan, ActionEnvelope


@pytest.fixture
def mock_config():
    return LiveBotConfig(
        symbol="BTCUSDT",
        testnet=True,
        dry_run=True,
        rosa_roja_enabled=True,
        max_position_pct=0.05,
        cooldown_ms=100,  # Corto para tests
        min_price_change_pct=0.0001,
        dynamic_cooldown=False,
        phi_moe_threshold=0.5,
        emergency_lambda_threshold=0.95,
        audit_log_path=None,
        state_snapshot_path=None,
    )


@pytest.fixture
def mock_engine():
    engine = MagicMock(spec=RosaRojaEngine)
    engine.process_event.return_value = ExecutionPlan(
        action="EXECUTE",
        chosen_trajectory=MagicMock(),
        global_confidence=0.75,
        envelope=ActionEnvelope(
            magnitude=0.8,
            bounds={"stop_pct": 0.02, "target_pct": 0.04},
            max_steps=10,
            metadata={"decision_trace": {"test": True}},
        ),
        invalidation_step=5,
        regime_alert=False,
        veto_details={},
    )
    return engine


@pytest.fixture
def mock_feed():
    feed = MagicMock(spec=BinanceWSFeed)
    feed.symbol = "BTCUSDT"
    feed.is_connected = True
    feed.state = "connected"
    feed.order_book = MagicMock()
    feed.order_book.is_initialized = True
    feed.order_book.mid_price = 50000.0
    feed.order_book.metrics = None
    return feed


@pytest.fixture
def mock_order_client():
    client = MagicMock(spec=BinanceOrderClient)
    client.submit_order = AsyncMock(return_value=MagicMock(
        order_id=12345,
        client_order_id="test_123",
        status="FILLED",
        side="BUY",
        executed_qty=0.001,
        avg_price=50000.0,
    ))
    client.cancel_all_orders = AsyncMock(return_value=0)
    client.close = AsyncMock()
    client.get_equity = AsyncMock(return_value=10000.0)
    return client


@pytest.fixture
def mock_account():
    account = MagicMock(spec=BinanceAccount)
    account.get_equity = AsyncMock(return_value=10000.0)
    account.get_position = AsyncMock(return_value=0.0)
    account.get_available_balance = AsyncMock(return_value=5000.0)
    account.start_auto_sync = AsyncMock()
    account.stop_auto_sync = AsyncMock()
    account.close = AsyncMock()
    return account


class TestLiveBotRunner:
    """Tests de integración para LiveBotRunner."""

    @pytest.mark.asyncio
    async def test_runner_initialization(self, mock_config, mock_engine, mock_feed, mock_order_client, mock_account):
        """Test que el runner se inicializa correctamente."""
        with patch('iot_machine_learning.infrastructure.adapters.market.live_runner.create_live_bot') as mock_create:
            runner = LiveBotRunner(mock_config)
            runner._engine = mock_engine
            runner._feed = mock_feed
            runner._order_client = mock_order_client
            runner._account = mock_account

            # Verificar componentes
            assert runner._engine is mock_engine
            assert runner._feed is mock_feed
            assert runner._order_client is mock_order_client
            assert runner._account is mock_account
            assert runner._state is not None

    @pytest.mark.asyncio
    async def test_can_execute_cooldown(self, mock_config, mock_engine, mock_feed):
        """Test que el cooldown previene ejecuciones muy seguidas."""
        runner = LiveBotRunner(mock_config)
        runner._state = LiveBotState(
            last_phi_moe=0.8,
            last_lambda_t=0.3,
        )
        runner._state.last_execution_time = time.time() - 0.05  # 50ms ago
        runner.config.cooldown_ms = 500
        runner.config.dynamic_cooldown = False

        # Mock plan
        plan = MagicMock()
        plan.action = "EXECUTE"
        plan.global_confidence = 0.75

        # No debe ejecutar (cooldown no cumplido)
        assert runner._can_execute(plan) is False

        # Avanzar tiempo
        runner._state.last_execution_time = time.time() - 1.0  # 1s ago
        assert runner._can_execute(plan) is True

    @pytest.mark.asyncio
    async def test_can_execute_price_change(self, mock_config):
        """Test que el cambio mínimo de precio se respeta."""
        runner = LiveBotRunner(mock_config)
        runner._state = LiveBotState(
            last_phi_moe=0.8,
            last_lambda_t=0.3,
        )
        runner._state.last_execution_time = 0  # No cooldown
        runner._state.last_execution_price = 50000.0
        runner.config.min_price_change_pct = 0.001  # 0.1%
        runner.config.dynamic_cooldown = False

        # Mock feed para obtener precio actual
        mock_feed = MagicMock()
        mock_feed.order_book = MagicMock()
        mock_feed.order_book.mid_price = 50005.0  # 0.01% change - menos que 0.1%

        runner._feed = mock_feed
        runner.config.min_price_change_pct = 0.001

        plan = MagicMock()
        plan.action = "EXECUTE"
        plan.global_confidence = 0.75

        # No debe ejecutar (cambio insuficiente)
        assert runner._can_execute(plan) is False

        # Con cambio suficiente
        mock_feed.order_book.mid_price = 50100.0  # 0.2% change
        assert runner._can_execute(plan) is True

    @pytest.mark.asyncio
    async def test_can_execute_phi_threshold(self, mock_config):
        """Test que phi_moe threshold se respeta."""
        runner = LiveBotRunner(mock_config)
        runner._state = LiveBotState()
        runner._state.last_phi_moe = 0.4  # Debajo de threshold 0.5
        runner._state.last_execution_time = 0
        runner.config.phi_moe_threshold = 0.5

        plan = MagicMock()
        plan.action = "EXECUTE"
        plan.global_confidence = 0.4

        assert runner._can_execute(plan) is False

        runner._state.last_phi_moe = 0.6
        assert runner._can_execute(plan) is True

    @pytest.mark.asyncio
    async def test_can_execute_lambda_threshold(self, mock_config):
        """Test que emergency_lambda_threshold bloquea ejecución."""
        runner = LiveBotRunner(mock_config)
        runner._state = LiveBotState()
        runner._state.last_lambda_t = 0.96  # > 0.95 threshold
        runner._state.last_execution_time = 0
        runner._state.last_phi_moe = 0.8
        runner.config.emergency_lambda_threshold = 0.95

        plan = MagicMock()
        plan.action = "EXECUTE"
        plan.global_confidence = 0.8

        assert runner._can_execute(plan) is False

        runner._state.last_lambda_t = 0.9
        assert runner._can_execute(plan) is True

    @pytest.mark.asyncio
    async def test_runner_shutdown(self, mock_config, mock_engine, mock_feed, mock_order_client, mock_account):
        """Test graceful shutdown."""
        runner = LiveBotRunner(mock_config)
        runner._engine = mock_engine
        runner._feed = mock_feed
        runner._order_client = mock_order_client
        runner._account = mock_account
        runner._running = True

        await runner.shutdown()

        assert runner._running is False
        mock_feed.disconnect.assert_called_once()
        mock_order_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_dynamic_cooldown(self, mock_config):
        """Test cooldown dinámico según lambda_t."""
        runner = LiveBotRunner(mock_config)
        runner.config.dynamic_cooldown = True
        runner.config.cooldown_ms = 500

        # lambda_t bajo -> cooldown normal
        assert runner.config.get_effective_cooldown(0.3) == 500
        assert runner.config.get_effective_cooldown(0.4) == 500

        # lambda_t medio -> cooldown reducido
        assert runner.config.get_effective_cooldown(0.6) == 250

        # lambda_t alto -> cooldown muy reducido (min 100ms)
        assert runner.config.get_effective_cooldown(0.9) == 100  # max(500*0.2, 100)
        assert runner.config.get_effective_cooldown(0.99) == 100

    @pytest.mark.asyncio
    async def test_state_persistence(self, mock_config, tmp_path):
        """Test guardado y carga de estado."""
        state_path = tmp_path / "state.json"
        mock_config.state_snapshot_path = tmp_path / "state.json"

        runner = LiveBotRunner(mock_config)
        runner._state = LiveBotState(
            cycle_count=10,
            last_execution_time=time.time(),
            last_execution_price=50000.0,
            last_execution_side="BUY",
            last_phi_moe=0.75,
            last_lambda_t=0.3,
            last_phi_ritmo=0.8,
            current_position=0.001,
            total_pnl=150.5,
            trades_count=5,
        )

        await runner._save_state()

        # Crear nuevo runner y cargar
        runner2 = LiveBotRunner(mock_config)
        runner2._state_path = tmp_path / "state.json"
        await runner2._load_state()

        assert runner2._state.cycle_count == 10
        assert runner2._state.last_execution_price == 50000.0
        assert runner2._state.last_execution_side == "BUY"
        assert runner2._state.trades_count == 5


# Test de integración con mocks de Binance (requiere Testnet real)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_bot_smoke_test():
    """Smoke test con Binance Testnet real (requiere API keys).
    
    Ejecutar con: pytest tests/integration/test_live_bot_runner.py -v -k integration --tb=short
    Requiere: BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_API_SECRET en env.
    """
    import os
    if not os.getenv("BINANCE_TESTNET_API_KEY") or not os.getenv("BINANCE_TESTNET_API_SECRET"):
        pytest.skip("Binance Testnet API keys not configured")

    config = LiveBotConfig(
        symbol="BTCUSDT",
        testnet=True,
        dry_run=True,
        cooldown_ms=1000,
        audit_log_path=None,
        state_snapshot_path=None,
        rosa_roja_enabled=True,
    )

    runner = await create_live_bot(config)
    
    # Solo conectar y verificar que el feed funciona
    await runner._feed.connect()
    assert runner._feed.is_connected
    
    # Procesar algunos eventos
    count = 0
    async for obs in runner._feed.iter_observations():
        count += 1
        if count >= 10:
            break
    
    assert count == 10
    await runner._feed.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])