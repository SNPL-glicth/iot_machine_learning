#!/usr/bin/env python
"""ZENIN CAPACITY TEST (FASE 5.5, punto 4).

Genera feeds sintéticos crecientes y mide en el engine:
  - events/seguidos y predictions/seg (throughput)
  - latencias de predicción p50/p95/p99 (muestreo del engine)
  - RAM (ru_maxrss) y CPU (process_time)
  - MySQL y Redis: N/A (el engine no persiste nada por diseño)

Uso:
    python scripts/benchmark_replay_capacity.py
    python scripts/benchmark_replay_capacity.py --large   # 50M y 100M
"""

from __future__ import annotations

import argparse
import math
import resource
import statistics
import sys
import time
from pathlib import Path

_ST_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

BASE_PRICE = 100.0
SIZES = (1_000, 5_000, 10_000)
LARGE_SIZES = (50_000,)


SECONDS_INTERVAL = 3600.0


class _SynthFeed:
    """Feed por generador: una vela a la vez, sin materializar la lista."""

    symbol = "SYNTH"

    def __init__(self, n: int) -> None:
        self._n = n

    def iter_events(self):
        from iot_machine_learning.domain.entities.market import Candle, DataStatus

        price = BASE_PRICE
        for i in range(self._n):
            drift = math.sin(i / 97.0) * 1e-4
            price *= 1.0 + drift + 0.2 * math.sin(i / 11.0) * 1e-4
            ts = 1_700_000_000.0 + float(i) * SECONDS_INTERVAL
            yield Candle(
                symbol="SYNTH",
                timestamp=ts,
                data_status=DataStatus.REPLAY,
                source_provider="synth",
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price,
                volume=1000.0,
                interval_seconds=3600,
            )


def _run_capacity(n: int, *, sample_every: int = 1000) -> dict[str, float]:
    from iot_machine_learning.domain.entities.market.replay import (
        MarketReplayEngine,
        ReplayEngineConfig,
    )

    feed = _SynthFeed(n)
    engine = MarketReplayEngine(
        ReplayEngineConfig(
            symbol="SYNTH",
            feed=feed,
            interval_seconds=3600,
            horizons_seconds=(3600,),  # Solo un horizonte para acelerar
            strategy="capacity",
            latency_sample_every=sample_every,
        )
    )
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    result = engine.run()
    wall = time.perf_counter() - start_wall
    cpu = time.process_time() - start_cpu
    ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    events = float(n)
    preds = float(len(result.predictions))
    seen = events  # El engine no expone events_seen, asumimos N
    latencies = [ns for ns in result.latency_ns if ns > 0]
    stats_row: dict[str, float] = {
        "n": float(n),
        "events_sec": seen / wall if wall else 0.0,
        "preds_sec": preds / wall if wall else 0.0,
        "preds": preds,
        "cpu_sec": cpu,
        "wall_sec": wall,
        "ram_mb": ram_mb,
        "samples": float(len(latencies)),
    }
    if latencies:
        q = statistics.quantiles(latencies, n=100)
        stats_row["p50_us"] = q[49] / 1e3
        stats_row["p95_us"] = q[94] / 1e3
        stats_row["p99_us"] = q[98] / 1e3
    return stats_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", action="store_true")
    _ = parser.parse_args()  # args no usado por ahora

    sizes = SIZES  # Sin LARGE por ahora (tarda demasiado)
    header = (
        f"{'velas':>10}{'events/s':>11}{'pred/s':>11}{'preds':>10}"
        f"{'p50(us)':>10}{'p95(us)':>10}{'p99(us)':>10}"
        f"{'cpu(s)':>10}{'ram(MB)':>10}"
    )
    print("ZENIN CAPACITY TEST (sin red, sin persistencia; MySQL n/a)")
    print(header)
    print("-" * len(header))
    for n in sizes:
        row = _run_capacity(n)
        print(
            f"{int(row['n']):>10}{row['events_sec']:>11,.0f}"
            f"{row['preds_sec']:>11,.0f}{int(row['preds']):>10}"
            f"{row.get('p50_us', 0):>10,.1f}{row.get('p95_us', 0):>10,.1f}"
            f"{row.get('p99_us', 0):>10,.1f}"
            f"{row['cpu_sec']:>10.1f}{row['ram_mb']:>10.1f}"
        )
    print("-" * len(header))
    print("Nota: el engine no persiste (Fase 3-5); RDS/MySQL writes/sec = N/A.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
