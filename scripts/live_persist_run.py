#!/usr/bin/env python
"""ZENIN LIVE PERSISTENCE RUN (FASE 7) — ZENIN con memoria.

MODO SHADOW + PERSISTENCIA: sin dinero real. El pipeline completo de la
FASE 7 sobre un fragmento live (LiveFeed + LiveClock):

    Live -> ZENIN -> Prediction -> MySQL (zenin_market)
                      -> espera horizonte -> OutcomeResolver
                      -> Outcome -> Evaluation -> Reward
                      -> Performance history (tablero de ZENIN)

Regla de FASE 7: ZENIN SOLO REGISTRA, NO APRENDE. El reward se guarda
para calibración posterior; ningún dato de esta fase modifica el modelo.
Futuro (fuera de esta fase): Reward history -> Calibration -> Expert
performance -> Strategy performance -> MoE adaptation.

El fragmento y las caídas se expresan en hora de mercado (09:30 =
apertura de la sesión), igual que en live_shadow_run.py.

Uso:
    python scripts/live_persist_run.py
    python scripts/live_persist_run.py --symbol NVDA --resolution 1m \\
        --start 09:30 --end 10:30
    python scripts/live_persist_run.py --drop 09:40:00-09:41:30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_ST_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV, override=True)

from iot_machine_learning.domain.entities.market.prediction.resolver import (  # noqa: E402
    OutcomeResolver,
)
from iot_machine_learning.domain.entities.market.replay import (  # noqa: E402
    LiveClock,
    ReplayEngineConfig,
)
from iot_machine_learning.infrastructure.adapters.market import (  # noqa: E402
    RESOLUTIONS,
    DropWindowsFeed,
    FragmentFeed,
    HistoricalCsvFeed,
    LiveFeed,
    LiveShadowRunner,
    drop_windows,
    fmt_ts,
    fragment_bounds,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market import (  # noqa: E402
    MarketPredictionRepository,
    ZeninMarketDbConnection,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.market_prediction_repository import (  # noqa: E402
    row_to_prediction,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "market"

BANNER = (
    "==================================================\n"
    "   ZENIN MARKET — LIVE PERSISTENCE (FASE 7)\n"
    "              NO REAL MONEY\n"
    "       NO ADAPTIVE LEARNING (SOLO REGISTRA)\n"
    "=================================================="
)


class CandlePriceLookup:
    """PriceLookup sobre el feed: último cierre a lo sumo en el plazo."""

    def __init__(self, feed: FragmentFeed | DropWindowsFeed) -> None:
        self._closes = tuple(
            (c.timestamp, c.close) for c in feed.iter_events()
        )

    def last_close(self, at_or_before: float) -> float | None:
        best: float | None = None
        for ts, close in self._closes:
            if ts <= at_or_before:
                best = close
            else:
                break
        return best


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _record_block(row: dict) -> str:
    """El record que ZENIN recuerda por predicción resuelta:
    14:30 NVDA 15m P(up)=0.71 expected=+0.82% conf=0.68
    actual=+1.17% direction=correct within=true reward=+X"""
    ts = fmt_ts(float(row["observation_timestamp"]))
    direction = "correct" if row["direction_correct"] else "incorrect"
    within = "true" if row["within_interval"] else "false"
    return (
        f"{ts} {row['symbol']} {row['horizon_seconds']}s "
        f"P(up)={row['probability_up']:.2f} "
        f"expected={_pct(row['expected_return'])} "
        f"conf={row['confidence']:.2f} "
        f"actual={_pct(row['outcome_return_realized'])} "
        f"direction={direction} within={within} "
        f"reward={row['reward_total']:+.4f} status={row['status']}"
    )


def _render_history(history: dict) -> str:
    lines: list[str] = []

    def _rows(group: str) -> list[str]:
        out: list[str] = []
        for r in history[group]:
            key = r["key"]
            if key is None:
                key = "-"
            n = r["evaluated"] or 0
            hits = r["hits"] or 0
            rate = f"{hits / n:.1%}" if n else "-"
            cal = (
                f"{r['calibration']:.3f}" if r["calibration"] is not None else "-"
            )
            out.append(
                f"  {key:<14} n={n:<4} dir_rate={rate:<7} "
                f"cal={cal:<7} reward={r['reward']:+.4f}"
            )
        return out

    lines.append("PERFORMANCE HISTORY (zenin_market)")
    lines.append("----------------------------------")
    lines.append("Por horizonte:")
    lines.extend(_rows("by_horizon") or ["  (sin datos)"])
    lines.append("Por estrategia:")
    lines.extend(_rows("by_strategy") or ["  (sin datos)"])
    lines.append("Por regimen:")
    lines.extend(_rows("by_regime") or ["  (sin datos)"])
    lines.append("Calibracion por bucket de confianza (P(up) declarada vs acierto):")
    lines.extend(
        (
            f"  bucket={r['bucket']:.1f} n={r['evaluated'] or 0} "
            f"hits={r['hits'] or 0} avg_p={r['avg_probability']:.3f}"
            for r in history["by_confidence"]
        )
        or ["  (sin datos)"]
    )
    lines.append("Series por dia:")
    lines.extend(
        (
            f"  {r['day']} n={r['evaluated'] or 0} "
            f"hits={r['hits'] or 0} reward={r['reward']:+.4f}"
            for r in history["by_day"]
        )
        or ["  (sin datos)"]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS), default="1m")
    parser.add_argument("--start", default="09:30", help="inicio fragmento (UTC)")
    parser.add_argument("--end", default="10:30", help="fin fragmento (UTC)")
    parser.add_argument("--drop", action="append", default=[],
                        help="ventana de caída HH:MM:SS-HH:MM:SS en hora de "
                             "mercado (repetible; 09:30 = apertura de sesión)")
    args = parser.parse_args()

    interval, horizons = RESOLUTIONS[args.resolution]
    path = DATA_DIR / f"{args.symbol}_{args.resolution}.csv"
    if not path.exists():
        print(f"dataset no existe: {path} (ejecuta scripts/download_market_data.py)")
        return 1

    print(BANNER)
    print()

    if not ZeninMarketDbConnection.health_check():
        print("MySQL zenin_market no disponible: revisa .env (MYSQL_*) y el "
              "contenedor (docker-compose.yml)")
        return 1

    from iot_machine_learning.infrastructure.persistence.sql.zenin_market.migrations import (  # noqa: E501
        apply_migrations,
    )

    full = HistoricalCsvFeed(path, symbol=args.symbol, interval_seconds=interval)
    all_events = tuple(full.iter_events())
    if not all_events:
        print(f"dataset vacío: {path}")
        return 1

    start_ts, end_ts = fragment_bounds(all_events[0].timestamp, args.start, args.end)
    fragment = FragmentFeed(full, start_ts, end_ts)
    if len(fragment) == 0:
        print(f"fragmento vacío: {args.start} -> {args.end} (UTC) sobre {path}")
        return 1

    drops = drop_windows(start_ts, args.drop)

    print("FASE 7 PIPELINE")
    print("---------------")
    print("Live -> ZENIN -> Prediction -> MySQL -> OutcomeResolver -> "
          "Outcome -> Evaluation -> Reward -> History")
    print(f"Fragmento: {fmt_ts(start_ts)} -> {fmt_ts(end_ts)} (UTC) | "
          f"instrumento: {args.symbol} | resolución: {interval}s | "
          f"horizontes: {horizons}")
    if drops:
        print("Simulated drops:")
        for start, end in drops:
            print(f"  - {fmt_ts(start)} -> {fmt_ts(end)} (UTC)")
    print()

    live_feed = LiveFeed(
        symbol=args.symbol,
        historical_feed=DropWindowsFeed(fragment, drops) if drops else fragment,
        expected_interval_seconds=interval,
    )
    shadow = LiveShadowRunner(
        live_feed,
        ReplayEngineConfig(
            symbol=args.symbol,
            feed=live_feed,
            interval_seconds=interval,
            horizons_seconds=horizons,
            initial_clock=LiveClock(now=start_ts),
        ),
    ).run()

    predictions = list(shadow.all_predictions)
    print(f"Predicciones emitidas: {len(predictions)}")
    print()

    apply_migrations()
    print()

    with ZeninMarketDbConnection.get_connection() as conn:
        repo = MarketPredictionRepository(conn)

        written = repo.save_batch(predictions)
        print(f"Persistidas en zenin_market.market_predictions: {written} "
              "(upsert idempotente por prediction_id)")
        print()

        resolver = OutcomeResolver()
        prices = CandlePriceLookup(DropWindowsFeed(fragment, drops) if drops else fragment)
        pending = repo.pending_outcomes(symbol=args.symbol)
        batch = resolver.resolve(
            (row_to_prediction(row) for row in pending), prices
        )
        resolved = list(batch.resolved)
        if resolved:
            repo.save_batch(resolved)
        print(f"OutcomeResolver: {batch.resolved_count} resueltas | "
              f"{batch.waiting_count} en espera de horizonte | "
              f"{len(batch.unchanged)} ya terminales")
        if not pending:
            print("  (el engine resuelve/invalida todo dentro del fragmento; en "
                  "operacion continua el resolver corre sobre las filas en espera "
                  "de corridas anteriores)")
        print()

        print("RECORD (ultimas resueltas: ZENIN recuerda)")
        print("------------------------------------------")
        shown = 0
        for row in repo.recent_records(symbol=args.symbol, status="rewarded"):
            print("  " + _record_block(row))
            shown += 1
        if not shown:
            print("  (ninguna resuelta aun: el horizonte no vencio en el fragmento)")
        print()

        history = repo.performance_history(symbol=args.symbol)
        print(_render_history(history))
        print()

    print("NO ADAPTIVE LEARNING: los rewards se guardan, no se aprenden.")
    print("Proxima etapa: Reward history -> Calibration -> Expert/Strategy "
          "performance -> MoE adaptation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
