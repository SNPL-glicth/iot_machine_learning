#!/usr/bin/env python
"""ZENIN LIVE SHADOW RUN (FASE 6) — ZENIN consumiendo un feed live.

MODO SHADOW: sin dinero real, sin persistencia. El mismo engine del
replay (Mismo objeto MarketObservation, mismo pipeline) recibe eventos
a través de LiveFeed + LiveClock: la diferencia con el replay es la
fuente y el reloj, nunca la lógica.

Condiciones de FASE 6:
    1. Live no altera el dominio (mismo MarketObservation → ZENIN).
    2. Reloj por Protocol Clock (ReplayClock vs LiveClock).
    3. Gaps visibles: GAP_DETECTED expected/received impresos.
    4. Estados de conexión explícitos (CONNECTED/DEGRADED/RECOVERED...).
    5. Sin persistencia: solo consola/dashboard (temporal).

El fragmento se expresa en la zona horaria del dataset (UTC).
Para simular datos perdidos: --drop 09:40:00-09:41:30 (repetible).

Uso:
    python scripts/live_shadow_run.py
    python scripts/live_shadow_run.py --symbol NVDA --resolution 1m \
        --start 09:30 --end 10:30
    python scripts/live_shadow_run.py --drop 09:40:00-09:41:30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ST_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

from iot_machine_learning.domain.entities.market.replay import (  # noqa: E402
    LiveClock,
    MarketReplayEngine,
    ReplayEngineConfig,
)
from iot_machine_learning.infrastructure.adapters.market import (  # noqa: E402
    RESOLUTIONS,
    DropWindowsFeed,
    FragmentFeed,
    HistoricalCsvFeed,
    LiveFeed,
    LiveShadowResult,
    LiveShadowRunner,
    drop_windows,
    fmt_ts,
    fragment_bounds,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "market"

BANNER = (
    "==================================================\n"
    "       ZENIN MARKET — LIVE SHADOW MODE\n"
    "              NO REAL MONEY\n"
    "=================================================="
)

_CORE_FIELDS = (
    "timestamp",
    "entry_price",
    "expected_return",
    "probability_up",
    "confidence",
    "horizon_seconds",
    "interval",
)


def _core(pred) -> tuple:
    return tuple(getattr(pred, f) for f in _CORE_FIELDS)


def _check_parity(fragment: FragmentFeed, drops: list[tuple[float, float]],
                  symbol: str, interval: int, horizons: tuple[int, ...]) -> None:
    """Live Shadow vs Replay sobre la misma secuencia: lógica idéntica."""
    replay_feed = fragment if not drops else DropWindowsFeed(fragment, drops)
    replay = MarketReplayEngine(
        ReplayEngineConfig(
            symbol=symbol,
            feed=replay_feed,
            interval_seconds=interval,
            horizons_seconds=horizons,
        )
    ).run()
    replay_map = {p.prediction_id: p for p in replay.predictions}

    live_feed = LiveFeed(
        symbol=symbol,
        historical_feed=DropWindowsFeed(fragment, drops) if drops else fragment,
        expected_interval_seconds=interval,
    )
    first = next(iter(fragment.iter_events()))
    shadow = LiveShadowRunner(
        live_feed,
        ReplayEngineConfig(
            symbol=symbol,
            feed=live_feed,
            interval_seconds=interval,
            horizons_seconds=horizons,
            initial_clock=LiveClock(now=first.timestamp),
        ),
    ).run()

    mismatches = 0
    for pred in shadow.all_predictions:
        counterpart = replay_map.get(pred.prediction_id)
        if counterpart is None or _core(pred) != _core(counterpart):
            mismatches += 1
    if mismatches:
        raise SystemExit(
            f"PARITY FAIL: {mismatches} predicciones divergen entre "
            "replay y live shadow (misma secuencia)"
        )
    print(f"PARITY OK: {len(shadow.all_predictions)} predicciones idénticas "
          f"(timestamp, features, prediction, horizon) replay vs live shadow")


def _render_result(result: LiveShadowResult, interval: int,
                   start_ts: float, end_ts: float, drops: list[tuple[float, float]]) -> str:
    lines: list[str] = []
    lines.append(f"Instrument: {result.symbol}")
    lines.append(f"Resolution: {interval}s")
    lines.append(f"Fragment:   {fmt_ts(start_ts)} -> {fmt_ts(end_ts)} (UTC)")
    lines.append(f"Predicciones: {len(result.all_predictions)} emitidas | "
                 f"{len(result.gaps)} gaps | estado final: "
                 f"{result.transitions[-1].state.value if result.transitions else 'CONNECTED'}")
    if drops:
        lines.append("Simulated drops:")
        for start, end in drops:
            lines.append(f"  - {fmt_ts(start)} -> {fmt_ts(end)} (UTC)")
    lines.append("")
    lines.append("CONNECTION STATE")
    lines.append("----------------")
    for transition in result.transitions:
        at = fmt_ts(transition.at_timestamp) if transition.at_timestamp is not None else "-"
        lines.append(f"  {at}  {transition.state.value}")
    lines.append("")
    if result.gaps:
        lines.append("GAP_DETECTED")
        lines.append("------------")
        for gap in result.gaps:
            lines.append(
                f"  expected: {fmt_ts(gap.expected_timestamp)}  "
                f"received: {fmt_ts(gap.received_timestamp)}  "
                f"gap: {gap.gap_seconds:.0f}s"
            )
    else:
        lines.append("GAP_DETECTED: ninguno (secuencia completa)")
    lines.append("")
    lines.append("PREDICTIONS")
    lines.append("-----------")
    lines.append(f"  Emitidas:          {len(result.all_predictions)}")
    lines.append(f"  INVALIDATED gap:   {len(result.invalidated_by_gap)} "
                 f"(reason=provider_gap)")
    for pred in result.invalidated_by_gap:
        lines.append(
            f"    {pred.prediction_id}  obs {fmt_ts(pred.observation.timestamp)}  "
            f"status={pred.status.value}  reason={pred.invalidation_reason}"
        )
    feed_end = [
        p for p in result.invalidated
        if p.invalidation_reason != "provider_gap"
    ]
    lines.append(f"  INVALIDATED fin:   {len(feed_end)} (horizonte sin vencer)")
    rewarded = sum(1 for p in result.predictions if p.status.value == "rewarded")
    lines.append(f"  REWARDED:          {rewarded}")
    lines.append("")
    lines.append("PERSISTENCE: OFF (temporal — FASE 6 primera prueba)")
    lines.append("  Live -> ZENIN -> console/dashboard")
    lines.append("  Siguiente etapa: Live -> Prediction -> MySQL -> Outcome -> Reward")
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
    parser.add_argument("--no-parity", action="store_true",
                        help="omite el chequeo Live Shadow vs Replay")
    args = parser.parse_args()

    interval, horizons = RESOLUTIONS[args.resolution]
    path = DATA_DIR / f"{args.symbol}_{args.resolution}.csv"
    if not path.exists():
        print(f"dataset no existe: {path} (ejecuta scripts/download_market_data.py)")
        return 1

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

    # Las ventanas de caída se expresan en hora de mercado (09:30 =
    # apertura de la sesión), igual que el fragmento.
    drops = drop_windows(start_ts, args.drop)

    print(BANNER)
    print()

    if not args.no_parity:
        _check_parity(fragment, drops, args.symbol, interval, horizons)
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

    print(_render_result(shadow, interval, start_ts, end_ts, drops))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
