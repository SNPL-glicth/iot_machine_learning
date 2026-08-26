#!/usr/bin/env python
"""ZENIN Live Bot — Entry point for live trading on Binance.

Usage:
    python scripts/zenin_live_btc.py --symbol BTCUSDT --testnet --dry-run
    python scripts/zenin_live_btc.py --config config/live_btc.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from iot_machine_learning.infrastructure.adapters.market.live_config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.live_runner import create_live_bot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ZENIN Live Bot — Event-driven trading on Binance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Basic config
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--testnet", action="store_true", default=True, help="Use Binance Testnet")
    parser.add_argument("--no-testnet", action="store_false", dest="testnet", help="Use Mainnet")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Simulate only, no real orders")
    parser.add_argument("--config", type=Path, help="Path to JSON config file")

    # Exchange settings
    parser.add_argument("--depth-speed", choices=["100ms", "1000ms"], default="100ms")
    parser.add_argument("--include-trades", action="store_true", default=True)
    parser.add_argument("--no-trades", action="store_false", dest="include_trades")
    parser.add_argument("--include-book-ticker", action="store_true", default=True)
    parser.add_argument("--no-book-ticker", action="store_false", dest="include_book_ticker")

    # Risk parameters
    parser.add_argument("--max-position-pct", type=float, default=0.05, help="Max position as % of equity")
    parser.add_argument("--max-lot-size", type=float, default=0.001, help="Max lot size (BTC)")
    parser.add_argument("--min-lot-size", type=float, default=0.00001, help="Min lot size (BTC)")
    parser.add_argument("--lot-size", type=float, default=0.00001, help="Lot size step")

    # Cooldown / Hysteresis
    parser.add_argument("--cooldown-ms", type=int, default=500, help="Min time between orders (ms)")
    parser.add_argument("--min-price-change-pct", type=float, default=0.0002, help="Min price change %")
    parser.add_argument("--dynamic-cooldown", action="store_true", default=True)
    parser.add_argument("--no-dynamic-cooldown", action="store_false", dest="dynamic_cooldown")

    # Decision thresholds
    parser.add_argument("--phi-moe-threshold", type=float, default=0.5, help="Phi_MoE threshold for EXECUTE")
    parser.add_argument("--geometric-threshold", type=float, default=-0.1, help="Cos(theta) threshold for EMERGENCY_FLUSH")
    parser.add_argument("--emergency-lambda", type=float, default=0.95, help="Lambda threshold for EMERGENCY_FLUSH")

    # Order types
    parser.add_argument("--use-post-only", action="store_true", default=True)
    parser.add_argument("--market-on-high-accel", action="store_true", default=True)
    parser.add_argument("--high-accel-threshold", type=float, default=0.8)

    # Stop/Target
    parser.add_argument("--stop-pct", type=float, default=0.02)
    parser.add_argument("--target-pct", type=float, default=0.04)

    # Emergency
    parser.add_argument("--emergency-lambda", type=float, default=0.95)
    parser.add_argument("--emergency-cancel-all", action="store_true", default=True)
    parser.add_argument("--emergency-close", action="store_true", default=True)

    # Connectivity
    parser.add_argument("--ws-reconnect-base", type=float, default=1.0)
    parser.add_argument("--ws-reconnect-max", type=float, default=60.0)
    parser.add_argument("--ws-ping-interval", type=float, default=20.0)
    parser.add_argument("--ws-ping-timeout", type=float, default=10.0)

    # Order book
    parser.add_argument("--ob-max-levels", type=int, default=100)
    parser.add_argument("--ob-sync-interval", type=float, default=30.0)

    # Persistence
    parser.add_argument("--audit-log-path", default="./logs/audit")
    parser.add_argument("--state-path", default="./data/state.json")
    parser.add_argument("--snapshot-interval", type=int, default=300)

    # Feature flags
    parser.add_argument("--audit-log", action="store_true", default=True)
    parser.add_argument("--no-audit-log", action="store_false", dest="enable_audit_log")
    parser.add_argument("--metrics-export", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--no-dry-run", action="store_false", dest="dry_run")

    # Rosa Roja
    parser.add_argument("--rosa-roja", action="store_true", default=True)
    parser.add_argument("--no-rosa-roja", action="store_false", dest="rosa_roja_enabled")
    parser.add_argument("--rosa-min-history", type=int, default=50)

    # Presets
    parser.add_argument("--preset", choices=["conservative", "aggressive", "testnet"], help="Predefined config preset")

    # Runtime
    parser.add_argument("--max-cycles", type=int, help="Max cycles (0 = infinite)")
    parser.add_argument("--max-runtime-min", type=int, help="Max runtime in minutes")
    parser.add_argument("--status-interval", type=int, default=10, help="Status log interval (seconds)")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> LiveBotConfig:
    """Construye config desde argumentos o archivo."""
    if args.config:
        return LiveBotConfig.from_file(args.config)

    # Presets
    if args.preset == "conservative":
        from iot_machine_learning.infrastructure.adapters.market.live_config import get_conservative_config
        config = get_conservative_config()
    elif args.preset == "aggressive":
        from iot_machine_learning.infrastructure.adapters.market.live_config import get_aggressive_config
        config = get_aggressive_config()
    elif args.preset == "testnet":
        from iot_machine_learning.infrastructure.adapters.market.live_config import get_testnet_config
        config = get_testnet_config()
    else:
        config = LiveBotConfig()

    # Override con argumentos CLI
    overrides = {
        "symbol": args.symbol,
        "testnet": args.testnet,
        "dry_run": args.dry_run,
        "depth_speed": args.depth_speed,
        "include_trades": args.include_trades,
        "include_book_ticker": args.include_book_ticker,
        "max_position_pct": args.max_position_pct,
        "max_lot_size": args.max_lot_size,
        "min_lot_size": args.min_lot_size,
        "lot_size": args.lot_size,
        "cooldown_ms": args.cooldown_ms,
        "min_price_change_pct": args.min_price_change_pct,
        "dynamic_cooldown": args.dynamic_cooldown,
        "phi_moe_threshold": args.phi_moe_threshold,
        "geometric_threshold": args.geometric_threshold,
        "emergency_lambda_threshold": args.emergency_lambda,
        "use_post_only": args.use_post_only,
        "market_on_high_accel": args.market_on_high_accel,
        "high_accel_threshold": args.high_accel_threshold,
        "default_stop_pct": args.stop_pct,
        "default_target_pct": args.target_pct,
        "emergency_lambda_threshold": args.emergency_lambda,
        "emergency_cancel_all": args.emergency_cancel_all,
        "emergency_close_position": args.emergency_close,
        "ws_reconnect_base_delay": args.ws_reconnect_base,
        "ws_reconnect_max_delay": args.ws_reconnect_max,
        "ws_ping_interval": args.ws_ping_interval,
        "ws_ping_timeout": args.ws_ping_timeout,
        "ob_max_levels": args.ob_max_levels,
        "ob_snapshot_interval_sec": args.ob_sync_interval,
        "audit_log_path": args.audit_log_path,
        "state_snapshot_path": args.state_path,
        "snapshot_interval_sec": args.snapshot_interval,
        "enable_audit_log": args.enable_audit_log,
        "enable_metrics_export": args.metrics_export,
        "dry_run": args.dry_run,
        "rosa_roja_enabled": args.rosa_roja_enabled,
        "rosa_roja_min_history": args.rosa_min_history,
    }

    # Aplicar solo valores no-None
    for k, v in overrides.items():
        if v is not None:
            setattr(config, k, v)

    return config


async def main() -> int:
    args = parse_args()
    config = build_config(args)

    # Setup logging level
    log_level = logging.DEBUG if config.dry_run else logging.INFO
    logging.getLogger().setLevel(log_level)

    # Banner
    print("=" * 60)
    print("  ZENIN LIVE BOT — Event-driven trading on Binance")
    print(f"  Symbol: {config.symbol} | Testnet: {config.testnet} | Dry-run: {config.dry_run}")
    print(f"  Rosa Roja: {'ON' if config.rosa_roja_enabled else 'OFF'} | Dry-run: {config.dry_run}")
    print("=" * 60)

    # Validate API keys for non-dry-run
    if not config.dry_run:
        import os
        if not os.getenv("BINANCE_API_KEY") and not os.getenv("BINANCE_TESTNET_API_KEY"):
            logger.error("BINANCE_API_KEY not set!")
            return 1
        if not os.getenv("BINANCE_API_SECRET") and not os.getenv("BINANCE_TESTNET_API_SECRET"):
            logger.error("BINANCE_API_SECRET not set!")
            return 1

    try:
        # Create and run bot
        runner = await create_live_bot(config)
        print(f"\nBot initialized. Starting event loop...")
        print("Press Ctrl+C to stop gracefully.\n")

        await runner.run()
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except Exception as e:
        logger.exception("Fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))