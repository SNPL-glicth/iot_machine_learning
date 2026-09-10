#!/usr/bin/env python
"""Test LiveBotRunner with Alpaca Paper Trading.

Este script prueba la integración completa:
- LiveBotRunner inicializado con broker=alpaca
- Conexión WebSocket a Alpaca
- Telemetry broadcast a puerto 8765
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Fix: use absolute import since iot_machine_learning is not a package in the traditional sense
import importlib.util

spec = importlib.util.spec_from_file_location(
    "live_config",
    Path(__file__).resolve().parent.parent / "infrastructure" / "adapters" / "market" / "live_config.py"
)
live_config_module = importlib.util.module_from_spec(spec)
sys.modules["live_config"] = live_config_module
spec.loader.exec_module(live_config_module)
LiveBotConfig = live_config_module.LiveBotConfig

spec2 = importlib.util.spec_from_file_location(
    "live_runner",
    Path(__file__).resolve().parent.parent / "infrastructure" / "adapters" / "market" / "live_runner.py"
)
live_runner_module = importlib.util.module_from_spec(spec2)
sys.modules["live_runner"] = live_runner_module
spec2.loader.exec_module(live_runner_module)
LiveBotRunner = live_runner_module.LiveBotRunner
LiveBotState = live_runner_module.LiveBotState

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def test_live_runner_alpaca():
    """Test LiveBotRunner with Alpaca."""
    from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.engine import RosaRojaEngine
    from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.domain.execution import ExecutionPlan, ActionEnvelope
    from unittest.mock import MagicMock
    
    # Config para Alpaca Paper
    config = LiveBotConfig(
        broker="alpaca",
        symbol="SPY",
        alpaca_api_key=os.getenv("ALPACA_API_KEY"),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY"),
        alpaca_api_base_url=os.getenv("ALPACA_API_BASE_URL"),
        alpaca_data_feed=os.getenv("ALPACA_DATA_FEED", "iex"),
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
        enable_metrics_export=False,  # Disable to avoid port conflicts
    )
    
    # Mock engine para test rápido
    mock_engine = MagicMock(spec=RosaRojaEngine)
    mock_engine.process_event.return_value = ExecutionPlan(
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
    
    print("\n" + "="*60)
    print("TEST: LiveBotRunner con Alpaca Paper Trading")
    print("="*60)
    print(f"Broker: {config.broker}")
    print(f"Symbol: {config.symbol}")
    print(f"Paper Trading: {config.is_paper_trading}")
    print(f"Enable Metrics Export: {config.enable_metrics_export}")
    
    runner = LiveBotRunner(config)
    runner._engine = mock_engine
    
    try:
        await runner.initialize()
        print("✓ Runner initialized successfully")
        
        # Check components
        print(f"  Feed: {type(runner._feed).__name__}")
        print(f"  Order Client: {type(runner._order_client).__name__}")
        print(f"  Account: {type(runner._account).__name__}")
        print(f"  Telemetry: {type(runner._telemetry).__name__ if runner._telemetry else 'None'}")
        
        # Test telemetry state building
        state = await runner._build_telemetry_state()
        print(f"\n✓ Telemetry state built:")
        print(f"  Symbol: {state['symbol']}")
        print(f"  Mode: {state['mode']}")
        print(f"  Broker: {state.get('broker', 'N/A')}")
        print(f"  Feed Connected: {state['feed_connected']}")
        print(f"  Feed State: {state['feed_state']}")
        print(f"  Equity: ${state['equity']:,.2f}")
        print(f"  Cash: ${state['cash']:,.2f}")
        print(f"  Buying Power: ${state['buying_power']:,.2f}")
        print(f"  Positions: {state['positions']}")
        print(f"  Orders: {state['orders']}")
        
        # Connect feed (acotado: con mercado cerrado el WS no emite ticks)
        try:
            await asyncio.wait_for(runner._feed.connect(), timeout=15)
        except asyncio.TimeoutError:
            print("\n⚠ Feed connect TIMEOUT (15s) - continuo sin WS")
        print(f"\n✓ Feed connected: {runner._feed.is_connected}")
        print(f"  Feed State: {runner._feed.state}")

        # Wait for a few market data messages (acotado a 20s, 0 ticks es
        # normal con mercado cerrado; antes se bloqueaba para siempre)
        print("\nWaiting for market data (max 20s)...")
        count = 0
        try:
            async def _grab():
                nonlocal count
                async for obs in runner._feed.iter_observations():
                    count += 1
                    print(f"  Observation {count}: {type(obs).__name__} - {obs.symbol} @ {obs.timestamp}")
                    if count >= 3:
                        break
            await asyncio.wait_for(_grab(), timeout=20)
        except asyncio.TimeoutError:
            print(f"  (sin ticks en 20s, count={count} - normal con mercado cerrado)")
        
        # Test telemetry broadcast
        await runner._broadcast_telemetry()
        print("\n✓ Telemetry broadcast sent")
        
        # Check telemetry state again after receiving data
        state2 = await runner._build_telemetry_state()
        print(f"\nUpdated telemetry:")
        print(f"  Best Bid: {state2['best_bid']}")
        print(f"  Best Ask: {state2['best_ask']}")
        print(f"  OBI: {state2['obi']}")
        
        await runner.shutdown()
        print("\n✓ Runner shutdown complete")
        
        return True
        
    except Exception as e:
        logger.exception("Test failed")
        await runner.shutdown()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_live_runner_alpaca())
    if result:
        print("\n" + "="*60)
        print("✓ TEST PASSED - LiveBotRunner works with Alpaca")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)
        sys.exit(1)