#!/usr/bin/env python
"""ZENIN MARKET RUN (FASE 5) — corre el replay sobre histórico real.

Imprime el marcador agregado del Market Replay para un instrumento.

Uso:
    python scripts/replay_market_run.py                      # NVDA 1m por defecto
    python scripts/replay_market_run.py --symbol NVDA --resolution 5m
    python scripts/replay_market_run.py --all                # 1m, 5m, 1h, 1d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ST_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "market"

RESOLUTIONS: dict[str, tuple[int, tuple[int, ...]]] = {
    "1m": (60, (60, 300, 900, 3600)),
    "5m": (300, (300, 900, 3600)),
    "1h": (3600, (3600,)),
    "1d": (86400, (86400,)),
}


def run_symbol(symbol: str, resolution: str) -> None:
    from iot_machine_learning.domain.entities.market.replay import (
        MarketReplayEngine,
        PerformanceReport,
        ReplayEngineConfig,
    )
    from iot_machine_learning.infrastructure.adapters.market import (
        HistoricalCsvFeed,
    )

    interval, horizons = RESOLUTIONS[resolution]
    path = DATA_DIR / f"{symbol}_{resolution}.csv"
    if not path.exists():
        print(f"dataset no existe: {path} (ejecuta scripts/download_market_data.py)")
        raise SystemExit(1)
    feed = HistoricalCsvFeed(path, symbol=symbol, interval_seconds=interval)
    engine = MarketReplayEngine(
        ReplayEngineConfig(
            symbol=symbol,
            feed=feed,
            interval_seconds=interval,
            horizons_seconds=horizons,
        )
    )
    result = engine.run()
    report = PerformanceReport.from_run(symbol, interval, result.predictions)
    print(report.render_ascii())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS), default="1m")
    parser.add_argument("--all", action="store_true", help="corre todas las resoluciones")
    args = parser.parse_args()

    if args.all:
        for resolution in RESOLUTIONS:
            try:
                run_symbol(args.symbol, resolution)
            except FileNotFoundError:
                continue
            print()
        return 0
    run_symbol(args.symbol, args.resolution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
