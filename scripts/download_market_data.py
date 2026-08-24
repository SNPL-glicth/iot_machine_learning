#!/usr/bin/env python
"""Descarga histórico real de mercado a disco (FASE 5 - etapa 0).

Herramienta de datos, NO runtime del dominio: se ejecuta una vez y
deja el histórico CONGELADO en ``data/market/``. El Market Replay jamás
toca la red: solo lee estos archivos.

Schema CSV por archivo (epoch float en segundos):
    ts_open,o,h,l,c,v,ts_close

Uso:
    python scripts/download_market_data.py --symbol NVDA
    python scripts/download_market_data.py --symbol AAPL --interval 5m --period 60d
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "market"

# interval -> periodo maximo servible por yfinance sin datos multi-ventana
INTERVAL_PERIODS: dict[str, str] = {
    "1m": "7d",
    "5m": "60d",
    "1h": "730d",
    "1d": "max",
}


def _download(symbol: str, interval: str, period: str, out: Path) -> int:
    try:
        import yfinance as yf  # dev tool, fuera del runtime del dominio
    except ImportError:
        print(
            "yfinance no está instalado: pip install yfinance",
            file=sys.stderr,
        )
        return 1

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        print(f"sin datos: {symbol} interval={interval} period={period}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ts_open", "o", "h", "l", "c", "v", "ts_close"])
        for ts, row in df.iterrows():
            writer.writerow(
                [
                    _epoch(ts),
                    _num(row["Open"]),
                    _num(row["High"]),
                    _num(row["Low"]),
                    _num(row["Close"]),
                    _num(row["Volume"]),
                    _epoch(ts) + _interval_seconds(interval),
                ]
            )
    print(f"{symbol} {interval}: {len(df)} velas -> {out}")
    return 0


def _epoch(ts: object) -> float:
    return float(ts.timestamp())


def _num(value: object) -> str:
    return f"{float(value):.6f}"


def _interval_seconds(interval: str) -> int:
    unit = interval[-1]
    n = int(interval[:-1])
    return n * {"m": 60, "h": 3600, "d": 86400}[unit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--interval", choices=list(INTERVAL_PERIODS), default="1m")
    parser.add_argument("--period", default=None)
    args = parser.parse_args()

    period = args.period or INTERVAL_PERIODS[args.interval]
    if args.interval == "1m":
        out = OUT_DIR / f"{args.symbol}_1m.csv"
    else:
        out = OUT_DIR / f"{args.symbol}_{args.interval}.csv"
    return _download(args.symbol, args.interval, period, out)


if __name__ == "__main__":
    raise SystemExit(main())
