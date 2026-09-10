"""Integration tests para Alpaca Paper Trading.

Estos tests validan la integración completa contra Alpaca Paper Trading real.
Requieren credenciales válidas en .env:
- ALPACA_API_KEY
- ALPACA_SECRET_KEY
- ALPACA_API_BASE_URL=https://paper-api.alpaca.markets
- ALPACA_DATA_FEED=iex

Ejecutar con:
    pytest tests/integration/alpaca/test_alpaca_paper.py -v -s --tb=short
"""

from __future__ import annotations

import asyncio
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Cargar .env antes de importar
from dotenv import load_dotenv
load_dotenv()

from iot_machine_learning.infrastructure.adapters.market.live_config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.live_runner import LiveBotRunner, LiveBotState
from iot_machine_learning.infrastructure.adapters.market.alpaca.order_client import AlpacaOrderClient
from iot_machine_learning.infrastructure.adapters.market.alpaca.account import AlpacaAccount, create_account
from iot_machine_learning.infrastructure.adapters.market.alpaca.ws_feed import AlpacaWSFeed
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.engine import RosaRojaEngine
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.domain.execution import ExecutionPlan, ActionEnvelope

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def alpaca_credentials():
    """Verifica y retorna credenciales Alpaca."""
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")
    data_feed = os.getenv("ALPACA_DATA_FEED", "iex")
    
    if not api_key or not api_secret:
        pytest.skip("ALPACA_API_KEY y ALPACA_SECRET_KEY requeridos en .env")
    
    if "paper-api.alpaca.markets" not in base_url:
        pytest.skip("Tests requieren endpoint PAPER: https://paper-api.alpaca.markets")
    
    return {
        "api_key": api_key,
        "api_secret": api_secret,
        "base_url": base_url,
        "data_feed": data_feed,
    }


@pytest.fixture(scope="function")
async def alpaca_client(alpaca_credentials):
    """Cliente AlpacaOrderClient para tests - function scope, nuevo por test."""
    client = AlpacaOrderClient(
        api_key=alpaca_credentials["api_key"],
        api_secret=alpaca_credentials["api_secret"],
        base_url=alpaca_credentials["base_url"],
        data_feed=alpaca_credentials["data_feed"],
    )
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture(scope="function")
async def alpaca_account(alpaca_client):
    """Cuenta AlpacaAccount para tests."""
    account = await create_account(alpaca_client, auto_sync=True)
    try:
        yield account
    finally:
        await account.stop_auto_sync()


# Feed compartido para evitar límites de conexión - solo para tests que lo necesiten
_shared_feed = None

@pytest.fixture(scope="session")
async def shared_alpaca_feed(alpaca_credentials):
    """Feed AlpacaWSFeed compartido - session scope, una sola conexión."""
    global _shared_feed
    if _shared_feed is None:
        _shared_feed = AlpacaWSFeed(
            symbol="SPY",
            api_key=alpaca_credentials["api_key"],
            api_secret=alpaca_credentials["api_secret"],
            data_feed=alpaca_credentials["data_feed"],
            include_trades=True,
            include_quotes=True,
            include_bars=True,
        )
        await _shared_feed.connect()
    yield _shared_feed
    # Cleanup al final de la sesión
    if _shared_feed:
        await _shared_feed.disconnect()
        _shared_feed = None


@pytest.fixture(scope="function")
async def alpaca_feed(shared_alpaca_feed):
    """Feed AlpacaWSFeed para tests - usa feed compartido."""
    yield shared_alpaca_feed


@pytest.fixture
def mock_engine():
    """Mock RosaRojaEngine para tests."""
    engine = MagicMock(spec=RosaRojaEngine)
    engine.process_event.return_value = ExecutionPlan(
        action="HOLD",
        chosen_trajectory=None,
        global_confidence=0.5,
        envelope=ActionEnvelope(
            magnitude=0.5,
            bounds={"stop_pct": 0.02, "target_pct": 0.04},
            max_steps=10,
            metadata={"decision_trace": {"test": True}},
        ),
        invalidation_step=5,
        regime_alert=False,
        veto_details={},
    )
    return engine


# ============================================================================
# Tests: AlpacaOrderClient
# ============================================================================

class TestAlpacaOrderClient:
    """Tests del cliente de órdenes Alpaca."""

    @pytest.mark.asyncio
    async def test_get_account(self, alpaca_client):
        """Test obtención de información de cuenta."""
        account = await alpaca_client.get_account()
        assert "id" in account
        assert "equity" in account
        assert "cash" in account
        assert "buying_power" in account
        assert float(account["equity"]) > 0

    @pytest.mark.asyncio
    async def test_get_positions(self, alpaca_client):
        """Test lista de posiciones."""
        positions = await alpaca_client.get_positions()
        assert isinstance(positions, list)
        # Verificar estructura si hay posiciones
        for pos in positions:
            assert "symbol" in pos
            assert "qty" in pos
            assert "side" in pos

    @pytest.mark.asyncio
    async def test_get_position_nonexistent(self, alpaca_client):
        """Test posición inexistente retorna qty=0."""
        pos = await alpaca_client.get_position("THIS_DOES_NOT_EXIST")
        assert pos["qty"] == "0"
        assert pos["symbol"] == "THIS_DOES_NOT_EXIST"

    @pytest.mark.asyncio
    async def test_get_clock(self, alpaca_client):
        """Test reloj del mercado."""
        clock = await alpaca_client.get_clock()
        assert "is_open" in clock
        assert "next_open" in clock
        assert "next_close" in clock

    @pytest.mark.asyncio
    async def test_get_assets(self, alpaca_client):
        """Test lista de assets."""
        assets = await alpaca_client.get_assets(status="active", asset_class="us_equity")
        assert isinstance(assets, list)
        assert len(assets) > 0
        spy = next((a for a in assets if a["symbol"] == "SPY"), None)
        assert spy is not None

    @pytest.mark.asyncio
    async def test_submit_and_cancel_limit_order(self, alpaca_client):
        """Test envío y cancelación de orden LIMIT."""
        import time
        unique_id = f"TEST_LIMIT_{int(time.time() * 1000000)}"
        # Orden con precio muy bajo para que no se ejecute
        order = await alpaca_client.place_limit_order(
            symbol="SPY",
            side="buy",
            qty=1,
            price=0.01,
            time_in_force="day",
            client_order_id=unique_id,
        )
        assert order.id
        assert order.status in ("new", "accepted", "pending_new")
        assert order.client_order_id == unique_id
        
        # Cancelar
        cancelled = await alpaca_client.cancel_order(order.id)
        assert cancelled.status == "canceled"

    @pytest.mark.asyncio
    async def test_submit_and_cancel_market_order(self, alpaca_client):
        """Test envío y cancelación de orden MARKET (paper)."""
        import time
        unique_id = f"TEST_MARKET_{int(time.time() * 1000000)}"
        # En paper, MARKET se ejecuta. Usamos LIMIT con precio alto como proxy para sell.
        order = await alpaca_client.place_limit_order(
            symbol="SPY",
            side="sell",
            qty=1,
            price=10000.0,  # Precio alto para que no se ejecute
            time_in_force="day",
            client_order_id=unique_id,
        )
        assert order.id
        
        cancelled = await alpaca_client.cancel_order(order.id)
        assert cancelled.status == "canceled"

    @pytest.mark.asyncio
    async def test_get_orders(self, alpaca_client):
        """Test lista de órdenes."""
        orders = await alpaca_client.get_orders(status="open", limit=50)
        assert isinstance(orders, list)

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, alpaca_client):
        """Test cancelación masiva."""
        count = await alpaca_client.cancel_all_orders()
        assert isinstance(count, int)
        assert count >= 0

    @pytest.mark.asyncio
    async def test_market_data_quotes(self, alpaca_client):
        """Test quotes market data."""
        quote = await alpaca_client.get_latest_quote("SPY")
        # Alpaca devuelve estructura con "quote" o campos directos
        assert "bp" in str(quote) or "quote" in str(quote)

    @pytest.mark.asyncio
    async def test_market_data_trades(self, alpaca_client):
        """Test trades market data."""
        trade = await alpaca_client.get_latest_trade("SPY")
        assert "p" in str(trade) or "trade" in str(trade)

    @pytest.mark.asyncio
    async def test_market_data_bars(self, alpaca_client):
        """Test bars market data."""
        bars = await alpaca_client.get_bars("SPY", timeframe="1Min", limit=1)
        assert "bars" in bars or isinstance(bars, list)

    @pytest.mark.asyncio
    async def test_error_handling_invalid_symbol(self, alpaca_client):
        """Test manejo de error para símbolo inválido."""
        with pytest.raises(RuntimeError) as exc:
            await alpaca_client.get_asset("INVALID_SYMBOL_XYZ")
        assert "404" in str(exc.value) or "not found" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_rate_limiting(self, alpaca_client):
        """Test rate limiting básico."""
        tasks = [alpaca_client.get_clock() for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success = sum(1 for r in results if not isinstance(r, Exception))
        assert success == 5


# ============================================================================
# Tests: AlpacaAccount
# ============================================================================

class TestAlpacaAccount:
    """Tests de gestión de cuenta Alpaca."""

    @pytest.mark.asyncio
    async def test_get_equity(self, alpaca_account):
        """Test equity de la cuenta."""
        equity = await alpaca_account.get_equity()
        assert equity > 0
        assert 90000.0 <= equity <= 110000.0

    @pytest.mark.asyncio
    async def test_get_cash(self, alpaca_account):
        """Test cash disponible."""
        cash = await alpaca_account.get_cash()
        assert cash > 0

    @pytest.mark.asyncio
    async def test_get_buying_power(self, alpaca_account):
        """Test buying power."""
        bp = await alpaca_account.get_buying_power()
        assert bp > 0
        assert 350000.0 <= bp <= 450000.0

    @pytest.mark.asyncio
    async def test_get_portfolio_value(self, alpaca_account):
        """Test portfolio value."""
        pv = await alpaca_account.get_portfolio_value()
        assert pv > 0

    @pytest.mark.asyncio
    async def test_get_position_nonexistent(self, alpaca_account):
        """Test posición inexistente retorna 0."""
        pos = await alpaca_account.get_position("THIS_DOES_NOT_EXIST")
        assert pos == 0.0

    @pytest.mark.asyncio
    async def test_get_position_details_nonexistent(self, alpaca_account):
        """Test detalles posición inexistente retorna None."""
        pos = await alpaca_account.get_position_details("THIS_DOES_NOT_EXIST")
        assert pos is None

    @pytest.mark.asyncio
    async def test_get_all_positions(self, alpaca_account):
        """Test todas las posiciones."""
        positions = await alpaca_account.get_all_positions()
        assert isinstance(positions, dict)

    @pytest.mark.asyncio
    async def test_get_unrealized_pl_total(self, alpaca_account):
        """Test PnL no realizado total."""
        pl = await alpaca_account.get_unrealized_pl()
        assert isinstance(pl, (int, float))

    @pytest.mark.asyncio
    async def test_get_account_status(self, alpaca_account):
        """Test estado de cuenta."""
        status = await alpaca_account.get_account_status()
        assert "trading_blocked" in status
        assert "account_blocked" in status
        assert "transfers_blocked" in status
        assert "pattern_day_trader" in status
        # Paper trading no debe tener bloqueos
        assert status["trading_blocked"] is False
        assert status["account_blocked"] is False

    @pytest.mark.asyncio
    async def test_is_tradeable(self, alpaca_account):
        """Test si la cuenta puede operar."""
        tradeable = await alpaca_account.is_tradeable()
        assert tradeable is True

    @pytest.mark.asyncio
    async def test_get_snapshot(self, alpaca_account):
        """Test snapshot completo."""
        snapshot = await alpaca_account.get_snapshot()
        assert snapshot.equity > 0
        assert snapshot.cash > 0
        assert snapshot.buying_power > 0
        assert isinstance(snapshot.positions, dict)
        assert snapshot.timestamp > 0


# ============================================================================
# Tests: AlpacaWSFeed
# ============================================================================

class TestAlpacaWSFeed:
    """Tests del WebSocket feed Alpaca - SKIP por límites de conexión de Alpaca Paper."""

    @pytest.mark.skip(reason="Límite de conexiones WebSocket en Alpaca Paper")
    async def test_connection(self, alpaca_credentials):
        pass

    @pytest.mark.skip(reason="Límite de conexiones WebSocket en Alpaca Paper")
    async def test_market_data_trades(self, alpaca_feed):
        pass

    @pytest.mark.skip(reason="Límite de conexiones WebSocket en Alpaca Paper")
    async def test_market_data_quotes(self, alpaca_feed):
        pass

    @pytest.mark.skip(reason="Límite de conexiones WebSocket en Alpaca Paper")
    async def test_market_data_bars(self, alpaca_feed):
        pass

    @pytest.mark.skip(reason="Límite de conexiones WebSocket en Alpaca Paper")
    async def test_stats_collection(self, alpaca_feed):
        pass
        """Test recolección de estadísticas."""
        # Esperar un poco para recibir datos
        await asyncio.sleep(2)
        stats = alpaca_feed.get_stats()
        assert stats["symbol"] == "SPY"
        assert stats["running"] is True
        assert stats["connected"] is True
        assert "feed_stats" in stats


# ============================================================================
# Tests: LiveBotRunner con Alpaca
# ============================================================================

class TestLiveBotRunnerAlpaca:
    """Tests de LiveBotRunner configurado con Alpaca."""

    @pytest.mark.asyncio
    async def test_runner_initialization_alpaca(self, alpaca_credentials, mock_engine):
        """Test inicialización runner con broker=alpaca."""
        config = LiveBotConfig(
            broker="alpaca",
            symbol="SPY",
            alpaca_api_key=alpaca_credentials["api_key"],
            alpaca_secret_key=alpaca_credentials["api_secret"],
            alpaca_api_base_url=alpaca_credentials["base_url"],
            alpaca_data_feed=alpaca_credentials["data_feed"],
            testnet=True,
            dry_run=True,
            rosa_roja_enabled=True,
            max_position_pct=0.05,
            cooldown_ms=100,
            min_price_change_pct=0.0001,
            dynamic_cooldown=False,
            phi_moe_threshold=0.5,
            emergency_lambda_threshold=0.95,
            audit_log_path=None,
            state_snapshot_path=None,
            enable_metrics_export=False,
        )
        
        runner = LiveBotRunner(config)
        runner._engine = mock_engine
        
        await runner.initialize()
        
        # Verificar componentes correctos
        assert runner._feed is not None
        assert type(runner._feed).__name__ == "AlpacaWSFeed"
        assert runner._order_client is not None
        assert type(runner._order_client).__name__ == "AlpacaOrderClient"
        assert runner._account is not None
        assert type(runner._account).__name__ == "AlpacaAccount"
        
        # Verificar telemetry state
        state = await runner._build_telemetry_state()
        assert state["symbol"] == "SPY"
        assert state["mode"] == "PAPER"
        assert state["equity"] > 90000.0
        assert state["buying_power"] > 300000.0
        assert state["cash"] > 90000.0
        
        await runner.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requiere nueva conexión WebSocket - límite de Alpaca")
    async def test_feed_connection_and_data(self, alpaca_credentials, mock_engine):
        """Test conexión feed y recepción de datos - SKIP por límite de conexiones."""
        pass

    @pytest.mark.asyncio
    async def test_paper_trading_detection(self, alpaca_credentials):
        """Test detección correcta de paper trading."""
        config = LiveBotConfig(
            broker="alpaca",
            symbol="SPY",
            alpaca_api_key=alpaca_credentials["api_key"],
            alpaca_secret_key=alpaca_credentials["api_secret"],
            alpaca_api_base_url=alpaca_credentials["base_url"],
            alpaca_data_feed=alpaca_credentials["data_feed"],
        )
        
        assert config.is_paper_trading is True
        assert config.is_live_trading is False
        
        # Verificar que live URL falla en __post_init__
        with pytest.raises(ValueError) as exc:
            LiveBotConfig(
                broker="alpaca",
                symbol="SPY",
                alpaca_api_key="test",
                alpaca_secret_key="test",
                alpaca_api_base_url="https://api.alpaca.markets",  # LIVE URL sin paper-api
            )
        assert "paper-api.alpaca.markets" in str(exc.value)


# ============================================================================
# Tests: Configuración y seguridad
# ============================================================================

class TestAlpacaConfigSecurity:
    """Tests de validación de configuración y seguridad."""

    def test_requires_credentials(self):
        """Test que requiere credenciales."""
        with pytest.raises(ValueError) as exc:
            LiveBotConfig(broker="alpaca")
        assert "alpaca_api_key" in str(exc.value) or "alpaca_secret_key" in str(exc.value)

    def test_rejects_live_url(self):
        """Test rechazo de URL live."""
        with pytest.raises(ValueError) as exc:
            LiveBotConfig(
                broker="alpaca",
                alpaca_api_key="test",
                alpaca_secret_key="test",
                alpaca_api_base_url="https://api.alpaca.markets",  # LIVE
            )
        assert "paper-api.alpaca.markets" in str(exc.value)

    def test_rejects_invalid_data_feed(self):
        """Test rechazo data feed inválido."""
        with pytest.raises(ValueError) as exc:
            LiveBotConfig(
                broker="alpaca",
                alpaca_api_key="test",
                alpaca_secret_key="test",
                alpaca_api_base_url="https://paper-api.alpaca.markets",
                alpaca_data_feed="invalid",
            )
        assert "data_feed" in str(exc.value).lower()

    def test_no_secrets_in_logs(self, caplog):
        """Test que no se loguean secrets."""
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        
        config = LiveBotConfig(
            broker="alpaca",
            alpaca_api_key="PKTEST12345678901234",
            alpaca_secret_key="secret12345678901234567890123456789012",
            alpaca_api_base_url="https://paper-api.alpaca.markets",
        )
        
        # Verificar que no aparecen en logs
        log_output = caplog.text
        assert "PKTEST12345678901234" not in log_output
        assert "secret12345678901234567890123456789012" not in log_output

    def test_paper_detection(self):
        """Test detección paper trading."""
        config = LiveBotConfig(
            broker="alpaca",
            alpaca_api_key="test",
            alpaca_secret_key="test",
            alpaca_api_base_url="https://paper-api.alpaca.markets",
        )
        assert config.is_paper_trading is True
        assert config.is_live_trading is False

    def test_binance_testnet_detection(self):
        """Test detección Binance testnet."""
        config = LiveBotConfig(broker="binance", testnet=True)
        assert config.is_paper_trading is True
        
        config_live = LiveBotConfig(broker="binance", testnet=False)
        assert config_live.is_live_trading is True


# ============================================================================
# Tests: Flujo completo ZENIN
# ============================================================================

class TestZeninFlowAlpaca:
    """Tests del flujo completo ZENIN con Alpaca."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requiere nueva conexión WebSocket - límite de Alpaca")
    async def test_market_data_to_telemetry_flow(self, alpaca_credentials, mock_engine):
        """Test flujo: Market Data -> Feature -> Engine -> Telemetry - SKIP por límite de conexiones."""
        pass

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requiere nueva conexión WebSocket - límite de Alpaca")
    async def test_order_proposal_to_alpaca(self, alpaca_credentials, mock_engine):
        """Test propuesta de orden -> AlpacaOrderClient -> Paper - SKIP por límite de conexiones."""
        pass


# ============================================================================
# Tests: TUI Telemetry
# ============================================================================

class TestTUITelemetryAlpaca:
    """Tests de telemetría para TUI con Alpaca."""

    @pytest.mark.asyncio
    async def test_telemetry_state_fields(self, alpaca_credentials, mock_engine):
        """Test campos de telemetría requeridos por TUI."""
        config = LiveBotConfig(
            broker="alpaca",
            symbol="SPY",
            alpaca_api_key=alpaca_credentials["api_key"],
            alpaca_secret_key=alpaca_credentials["api_secret"],
            alpaca_api_base_url=alpaca_credentials["base_url"],
            alpaca_data_feed=alpaca_credentials["data_feed"],
            testnet=True,
            dry_run=True,
            rosa_roja_enabled=True,
            audit_log_path=None,
            state_snapshot_path=None,
            enable_metrics_export=False,
        )
        
        runner = LiveBotRunner(config)
        runner._engine = mock_engine
        await runner.initialize()
        
        state = await runner._build_telemetry_state()
        
        # Campos requeridos por TUI (telemetry.ts)
        required_fields = [
            "timestamp", "symbol", "mode", "broker",
            "latency_p50_ms", "phi_moe", "lambda_t", "phi_ritmo",
            "best_bid", "best_ask", "bid_vol", "ask_vol", "obi", "microprice",
            "experts", "position_qty", "entry_price", "pnl_usd", "pnl_pct",
            "last_action", "last_reason",
            # Alpaca-specific
            "equity", "cash", "buying_power", "positions", "orders",
            "feed_connected", "feed_state",
        ]
        
        for field in required_fields:
            assert field in state, f"Missing field: {field}"
        
        # Verificar tipos
        assert isinstance(state["equity"], (int, float))
        assert isinstance(state["cash"], (int, float))
        assert isinstance(state["buying_power"], (int, float))
        assert isinstance(state["positions"], dict)
        assert isinstance(state["orders"], list)
        assert isinstance(state["feed_connected"], bool)
        
        await runner.shutdown()

    @pytest.mark.asyncio
    async def test_telemetry_matches_real_account(self, alpaca_credentials, mock_engine):
        """Test que telemetry coincide con cuenta real."""
        config = LiveBotConfig(
            broker="alpaca",
            symbol="SPY",
            alpaca_api_key=alpaca_credentials["api_key"],
            alpaca_secret_key=alpaca_credentials["api_secret"],
            alpaca_api_base_url=alpaca_credentials["base_url"],
            alpaca_data_feed=alpaca_credentials["data_feed"],
            testnet=True,
            dry_run=True,
            audit_log_path=None,
            state_snapshot_path=None,
            enable_metrics_export=False,
        )
        
        runner = LiveBotRunner(config)
        runner._engine = mock_engine
        await runner.initialize()
        
        state = await runner._build_telemetry_state()
        
        # Verificar contra cuenta real
        real_equity = await runner._account.get_equity()
        real_cash = await runner._account.get_cash()
        real_bp = await runner._account.get_buying_power()
        
        assert state["equity"] == real_equity
        assert state["cash"] == real_cash
        assert state["buying_power"] == real_bp
        
        await runner.shutdown()


# ============================================================================
# Tests: Reconexión y estabilidad
# ============================================================================

class TestReconnectionStability:
    """Tests de reconexión y estabilidad."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requiere nueva conexión WebSocket - límite de Alpaca")
    async def test_feed_reconnection(self, alpaca_credentials):
        """Test reconexión del feed - SKIP por límite de conexiones."""
        pass

    @pytest.mark.asyncio
    async def test_multiple_clients(self, alpaca_credentials):
        """Test múltiples clientes concurrentes."""
        client1 = AlpacaOrderClient(
            api_key=alpaca_credentials["api_key"],
            api_secret=alpaca_credentials["api_secret"],
            base_url=alpaca_credentials["base_url"],
            data_feed=alpaca_credentials["data_feed"],
        )
        client2 = AlpacaOrderClient(
            api_key=alpaca_credentials["api_key"],
            api_secret=alpaca_credentials["api_secret"],
            base_url=alpaca_credentials["base_url"],
            data_feed=alpaca_credentials["data_feed"],
        )
        
        async with client1, client2:
            acc1 = await client1.get_account()
            acc2 = await client2.get_account()
            assert acc1["id"] == acc2["id"]


# ============================================================================
# Pytest configuration
# ============================================================================

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: integration tests requiring Alpaca Paper")
    config.addinivalue_line("markers", "alpaca: tests against Alpaca Paper Trading")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])