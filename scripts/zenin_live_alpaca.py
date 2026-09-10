#!/usr/bin/env python
"""ZENIN Live Alpaca Paper Trading Bot.

Main entry point for live algorithmic trading on Alpaca (SPY, AAPL, etc.)
powered by Rosa Roja MoE Engine, Kalman/Taylor/Statistical jury, and
real-time telemetry broadcasting on port 8765 to ZENIN-TUI.

Usage:
    python scripts/zenin_live_alpaca.py --dry-run
    python scripts/zenin_live_alpaca.py --symbol SPY --no-dry-run
    python scripts/zenin_live_alpaca.py --config config/live_alpaca_paper.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project and parent ST root to sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_ST_ROOT = _PROJECT_ROOT.parent

for p in (str(_ST_ROOT), str(_PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv  # noqa: E402

# Load .env from project root or ST root
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_ST_ROOT / ".env")

from iot_machine_learning.infrastructure.adapters.market.live_config import (  # noqa: E402
    LiveBotConfig,
)
from iot_machine_learning.infrastructure.adapters.market.live_runner import (  # noqa: E402
    create_live_bot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("zenin.alpaca")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ZENIN Live Alpaca Bot - Event-Driven Market Trading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="SPY", help="Default/primary ticker symbol (e.g., SPY)")
    parser.add_argument("--symbols", default="SPY,QQQ,NVDA,AAPL", help="Comma-separated basket of symbols to trade concurrently")
    parser.add_argument("--max-positions", type=int, default=2, help="Max concurrent active positions across portfolio")
    parser.add_argument("--max-portfolio-exposure", type=float, default=0.25, help="Max total portfolio exposure fraction of equity")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Simulate execution without sending orders")
    parser.add_argument("--no-dry-run", action="store_false", dest="dry_run", help="Enable actual paper order execution")
    parser.add_argument("--config", type=Path, help="Path to JSON configuration file")
    parser.add_argument("--lot-size", type=float, default=1.0, help="Order lot size (shares)")
    parser.add_argument("--max-lot-size", type=float, default=25.0, help="Max lot size per order (shares)")
    parser.add_argument("--min-lot-size", type=float, default=1.0, help="Min lot size per order (shares)")
    parser.add_argument("--max-position-pct", type=float, default=0.10, help="Max position fraction of equity per trade")
    parser.add_argument("--cooldown-ms", type=int, default=500, help="Cooldown between orders in ms")
    parser.add_argument("--phi-moe-threshold", type=float, default=0.40, help="Phi_MoE execution threshold (intermediate balance)")
    parser.add_argument("--geometric-threshold", type=float, default=-0.4, help="Min cos theta before direction reversal abort")
    parser.add_argument("--trailing-activation", type=float, default=4.50, help="Trailing profit activation PnL in USD")
    parser.add_argument("--trailing-min-giveback", type=float, default=1.80, help="Min giveback in USD before trailing lock")
    parser.add_argument("--trailing-giveback-ratio", type=float, default=0.30, help="Giveback ratio from peak profit")
    parser.add_argument("--feed", default=os.getenv("ALPACA_DATA_FEED", "iex"), choices=["iex", "sip"], help="Alpaca data feed")
    parser.add_argument("--enforce-market-hours", action="store_true", default=False, help="Strictly flush and halt on 16:00 ET close (default: False for paper/extended)")
    parser.add_argument("--metrics-export", action="store_true", default=True, help="Broadcast WebSocket telemetry on 8765")
    parser.add_argument("--no-metrics-export", action="store_false", dest="metrics_export")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> LiveBotConfig:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets/v2")

    if not api_key or not secret_key:
        logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment or .env!")
        sys.exit(1)

    symbols_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else [args.symbol.upper()]
    primary_symbol = symbols_list[0] if symbols_list else args.symbol.upper()

    if args.config and args.config.exists():
        cfg = LiveBotConfig.from_file(args.config)
    else:
        cfg = LiveBotConfig(
            broker="alpaca",
            symbol=primary_symbol,
            symbols=symbols_list,
            max_concurrent_positions=args.max_positions,
            max_portfolio_exposure_pct=args.max_portfolio_exposure,
            testnet=True,
            dry_run=args.dry_run,
            alpaca_api_key=api_key,
            alpaca_secret_key=secret_key,
            alpaca_api_base_url=base_url,
            alpaca_data_feed=args.feed,
            max_lot_size=args.max_lot_size,
            min_lot_size=args.min_lot_size,
            lot_size=args.lot_size,
            max_position_pct=args.max_position_pct,
            cooldown_ms=args.cooldown_ms,
            phi_moe_threshold=args.phi_moe_threshold,
            enable_metrics_export=args.metrics_export,
            rosa_roja_enabled=True,
        )

    # Always ensure credentials and symbol overrides
    cfg.broker = "alpaca"
    cfg.symbol = primary_symbol
    cfg.symbols = symbols_list
    cfg.max_concurrent_positions = args.max_positions
    cfg.max_portfolio_exposure_pct = args.max_portfolio_exposure
    cfg.alpaca_api_key = api_key
    cfg.alpaca_secret_key = secret_key
    cfg.alpaca_api_base_url = base_url
    cfg.alpaca_data_feed = args.feed
    cfg.dry_run = args.dry_run
    cfg.enable_metrics_export = args.metrics_export
    cfg.phi_moe_threshold = args.phi_moe_threshold
    cfg.geometric_threshold = getattr(args, "geometric_threshold", -0.4)
    cfg.max_position_pct = args.max_position_pct
    cfg.max_lot_size = args.max_lot_size
    cfg.trailing_activation_pnl = args.trailing_activation
    cfg.trailing_min_giveback = args.trailing_min_giveback
    cfg.trailing_giveback_ratio = args.trailing_giveback_ratio
    cfg.enforce_market_hours = args.enforce_market_hours
    return cfg


async def main() -> int:
    args = parse_args()
    config = build_config(args)

    symbols_str = ", ".join(config.symbols) if config.symbols else config.symbol
    print("=" * 65)
    print("  🏛️  ZENIN LIVE MARKET BOT — ALPACA MULTI-ASSET PORTFOLIO")
    print(f"  Basket: [{symbols_str}] | Max Concurrent Pos: {config.max_concurrent_positions}")
    print(f"  Feed: {config.alpaca_data_feed} | Mode: {'DRY-RUN' if config.dry_run else 'LIVE PAPER'}")
    print(f"  Rosa Roja: {'ON' if config.rosa_roja_enabled else 'OFF'} | Telemetry WS: {'ws://127.0.0.1:8765' if config.enable_metrics_export else 'OFF'}")
    print(f"  Lot size: {config.lot_size} sh | Pos Pct: {config.max_position_pct * 100:.0f}% | Max Port: {config.max_portfolio_exposure_pct * 100:.0f}%")
    print("=" * 65)

    try:
        runner = await create_live_bot(config)
        logger.info("Bot initialized. Starting live feed and Rosa Roja event loop...")
        await runner.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Graceful shutdown complete.")
        return 0
    except Exception as e:
        logger.exception(f"Fatal error in Live Alpaca Bot: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
