"""Institutional CLI and unified startup manager for Zephyr Trading System."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from iot_machine_learning.infrastructure.adapters.market.zephyr.config import (
    ALPACA_PAPER_CONFIG_PATH,
    ZEPHYR_CONFIG_PATH,
    LiveBotConfig,
    load_zephyr_config,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.runners import (
    LiveBotRunner,
    create_live_bot,
)

logger = logging.getLogger("zephyr.cli")


def _setup_environment() -> None:
    """Loads environment files from workspace roots."""
    cur = Path(__file__).resolve().parent
    candidates = [
        cur.parents[4] / ".env",  # ST root
        cur.parents[3] / ".env",  # iot_machine_learning root
        cur / "config" / ".env",
    ]
    for c in candidates:
        if c.exists():
            load_dotenv(c)


def build_parser() -> argparse.ArgumentParser:
    """Builds institutional CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="zephyr",
        description="Zephyr High-Frequency & Event-Driven Algorithmic Trading Platform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--broker", choices=["alpaca", "binance"], default="alpaca", help="Target broker")
    parser.add_argument("--symbol", default=None, help="Primary trading ticker (e.g. SPY, BTCUSDT)")
    parser.add_argument("--symbols", default=None, help="Comma-separated basket of symbols (e.g. SPY,QQQ)")
    parser.add_argument("--config", type=Path, default=None, help="Path to configuration JSON profile")
    parser.add_argument("--secrets", type=Path, default=None, help="Path to local secrets JSON file")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Simulation mode without real orders")
    parser.add_argument("--testnet", action="store_true", default=True, help="Enable testnet / paper environment")
    parser.add_argument("--status", action="store_true", default=False, help="Inspect broker account & position status")
    return parser


async def inspect_status(config: LiveBotConfig) -> None:
    """Prints current broker account metrics and open positions."""
    from iot_machine_learning.infrastructure.adapters.market.zephyr.execution import (
        create_account,
        create_order_client,
    )

    client = create_order_client(config)
    account = create_account(config, client)
    try:
        equity = float(await account.get_equity()) if hasattr(account, "get_equity") else 0.0
        print(f"\n[ZEPHYR STATUS] Broker: {config.broker.upper()} | Mode: {'PAPER' if config.is_paper_trading else 'LIVE'}")
        print(f"Equity: ${equity:,.2f} USD")
        if hasattr(client, "get_positions"):
            positions = await client.get_positions()
            print(f"Open Positions ({len(positions)}):")
            for p in positions:
                print(f"  - {p.symbol}: {p.qty} shares/units @ ${getattr(p, 'avg_entry_price', 0.0):.2f}")
    finally:
        if hasattr(client, "close"):
            res = client.close()
            if asyncio.iscoroutine(res):
                await res


async def run_zephyr(args: argparse.Namespace) -> None:
    """Initializes and runs the sovereign Zephyr trading bot."""
    _setup_environment()

    # Load configuration
    cfg_path = args.config or (ALPACA_PAPER_CONFIG_PATH if args.broker == "alpaca" else ZEPHYR_CONFIG_PATH)
    config = load_zephyr_config(config_path=cfg_path, secrets_path=args.secrets)

    # Apply CLI overrides
    if args.broker:
        config.broker = args.broker
    if args.symbol:
        config.symbol = args.symbol
    if args.symbols:
        config.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    config.dry_run = args.dry_run
    config.testnet = args.testnet

    if args.status:
        await inspect_status(config)
        return

    logger.info("Initializing Zephyr on broker=%s with symbol=%s...", config.broker, config.symbol)
    runner = await create_live_bot(config)
    logger.info("Starting Zephyr sovereign live event loop...")
    await runner.run()


def main() -> None:
    """CLI execution entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(run_zephyr(args))
    except KeyboardInterrupt:
        logger.info("Zephyr terminated cleanly by operator (SIGINT).")
    except Exception as e:
        logger.critical("Fatal Zephyr startup error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
