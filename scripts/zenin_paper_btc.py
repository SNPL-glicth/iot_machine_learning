#!/usr/bin/env python
"""ZENIN PAPER BOT — Trading MVP 0.1 (BTC-USD, papel, $0).

BINANCE LIVE (klines públicas, sin API key)
     ↓
MarketObservation → FeatureWindow
     ↓
raw predictor (naive/momentum/ema-crossover)
     ↓
Calibration wrapper (FASE 10.5: prob_raw + prob_calibrated + versión)
     ↓
Evidence Gate (NO_TRADE / LONG / SHORT en papel)
     ↓
Prediction + Evidence → MySQL (zenin_market)
     ↓
OutcomeResolver → Evaluation → Reward
     ↓
Status line por ciclo (uptime, obs, preds, NO-TRADE rate, cal, conn)

Sin aprendizaje adaptativo en vivo. Sin dinero real. Sin órdenes.
La pregunta que responde este experimento: ¿qué hace ZENIN cuando tiene
que mirar un mercado real durante horas o días?

Uso:
    python scripts/zenin_paper_btc.py                       # 1m, 24/7
    python scripts/zenin_paper_btc.py --interval 300        # velas 5m
    python scripts/zenin_paper_btc.py --predictor naive
    python scripts/zenin_paper_btc.py --calibrator artifacts/calibrator.json
    python scripts/zenin_paper_btc.py --max-cycles 3        # smoke test

Requiere MySQL zenin_market (.env). Ctrl+C para apagar limpio.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_ST_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV, override=True)

BANNER = """
==================================================
       ZENIN MARKET — PAPER TRADING
                 BTC-USD
             NO REAL MONEY
==================================================
   Presupuesto del experimento: $0 COP
   Sin órdenes. Sin aprendizaje adaptativo.
==================================================
"""

INTERVAL_CHOICES = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}
HORIZONS_BY_INTERVAL = {
    60: (60, 300, 900),
    180: (180, 900),
    300: (300, 900, 1800),
    900: (900, 1800),
    1800: (1800,),
    3600: (3600,),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC-USD")
    parser.add_argument("--interval", choices=sorted(INTERVAL_CHOICES), default="1m")
    parser.add_argument(
        "--horizons",
        default=None,
        help="horizontes en segundos separados por coma "
        f"(default según intervalo: {HORIZONS_BY_INTERVAL})",
    )
    parser.add_argument(
        "--predictor",
        choices=["momentum", "naive", "ema-crossover"],
        default="momentum",
    )
    parser.add_argument(
        "--neutral-margin",
        type=float,
        default=0.05,
        help="semi-ancho de la zona neutral alrededor de 0.5",
    )
    parser.add_argument(
        "--allow-raw",
        action="store_true",
        help="operar señales UNCALIBRATED (solo experimentos controlados)",
    )
    parser.add_argument(
        "--calibrator",
        default=None,
        help="artefacto JSON del calibrador aceptado (export_calibrator_state)",
    )
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="correr sin MySQL (status line only; NO persiste el experimento)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    interval_seconds = INTERVAL_CHOICES[args.interval]
    horizons = (
        tuple(int(h) for h in args.horizons.split(","))
        if args.horizons
        else HORIZONS_BY_INTERVAL[interval_seconds]
    )

    print(BANNER)
    print(
        f"symbol={args.symbol} interval={args.interval} "
        f"horizons={horizons} predictor={args.predictor}"
    )
    print()

    # ── dependencias reales ─────────────────────────────────────────────
    from iot_machine_learning.infrastructure.adapters.market.binance_klines_feed import (
        BinanceKlinesFeed,
    )
    from iot_machine_learning.infrastructure.adapters.market.paper_runner import (
        PaperBotConfig,
        PaperBotRunner,
    )

    config = PaperBotConfig(
        symbol=args.symbol,
        interval_seconds=interval_seconds,
        horizons_seconds=horizons,
        predictor_name=args.predictor,
        neutral_margin=args.neutral_margin,
        require_calibrated=not args.allow_raw,
    )

    prediction_repo_factory = evidence_repo_factory = None
    if not args.no_db:
        from iot_machine_learning.infrastructure.persistence.sql.zenin_market import (
            CalibrationEvidenceRepository,
            MarketPredictionRepository,
            ZeninMarketDbConnection,
        )
        from iot_machine_learning.infrastructure.persistence.sql.zenin_market.migrations import (
            apply_migrations,
        )

        if not ZeninMarketDbConnection.health_check():
            print("MySQL zenin_market no disponible: revisa .env (MYSQL_*).")
            print("Tip: --no-db corre sin persistencia (no recomendado).")
            return 1
        apply_migrations()

        def prediction_repo_factory():
            with ZeninMarketDbConnection.get_connection() as conn:
                return MarketPredictionRepository(conn)

        def evidence_repo_factory():
            with ZeninMarketDbConnection.get_connection() as conn:
                return CalibrationEvidenceRepository(conn)

    if prediction_repo_factory is None:
        print("--no-db activo: el experimento NO será reproducible.")
        print()
        prediction_repo_factory = lambda: _NullPredictionRepo()
        evidence_repo_factory = lambda: _NullEvidenceRepo()

    feed = BinanceKlinesFeed(
        symbol=args.symbol,
        interval_seconds=interval_seconds,
    )
    runner = PaperBotRunner(
        config=config,
        feed=feed,
        prediction_repo_factory=prediction_repo_factory,
        evidence_repo_factory=evidence_repo_factory,
        calibrator_state_path=Path(args.calibrator) if args.calibrator else None,
    )

    stop_requested = {"flag": False}

    def _handle_sigint(signum, frame):  # noqa: ARG001
        stop_requested["flag"] = True
        print("\n[shutdown] señal recibida; cerrando tras este ciclo...")

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    try:
        while not stop_requested["flag"]:
            started = time.monotonic()
            report = runner.run_cycle()
            elapsed = time.monotonic() - started
            remaining = max(1.0, interval_seconds - elapsed)
            end = started + remaining
            while time.monotonic() < end and not stop_requested["flag"]:
                time.sleep(min(1.0, max(0.05, end - time.monotonic())))
            if args.max_cycles is not None and runner.cycle_count >= args.max_cycles:
                break
    finally:
        print()
        print("[shutdown] experimento detenido con limpieza.")
        print(f"           ciclos={runner.cycle_count}")
        print(f"           acciones acumuladas={dict(runner.totals)}")
    return 0


class _NullPredictionRepo:
    """Sustituto --no-db: descarta y reporta cero pendientes."""

    def save_batch(self, predictions):
        return len(list(predictions))

    def pending_outcomes(self, *, symbol=None):
        return []


class _NullEvidenceRepo:
    def save_batch(self, records):
        return len(list(records))


if __name__ == "__main__":
    raise SystemExit(main())
