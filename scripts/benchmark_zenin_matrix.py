#!/usr/bin/env python
"""ZENIN BENCHMARK (FASE 5.5) — baselines, walk-forward y performance matrix.

Imprime:
    1. Baselines en NVDA 1h (vs Naive): direction, brier, reward.
    2. Walk-forward TRAIN->TEST (momentum ajustado solo sobre TRAIN)
       vs Naive y Random sobre los mismos TEST.
    3. ZENIN PERFORMANCE MATRIX: direction % por régimen x horizonte,
       sobre todos los instrumentos descargados.

Uso:
    python scripts/benchmark_zenin_matrix.py
    python scripts/benchmark_zenin_matrix.py --symbols NVDA,AAPL,BTC-USD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ST_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "market"

DEFAULT_SYMBOLS = "NVDA,AAPL,MSFT,SPY,QQQ,BTC-USD,ETH-USD"
HORIZONS_1H = (3600, 14400, 86400)  # 1h, 4h, 1d
SEGMENT_VELAS = 200
REGIME_LOOKBACK = 120


def _feed(symbol: str, resolution: str):
    from iot_machine_learning.infrastructure.adapters.market import HistoricalCsvFeed

    return HistoricalCsvFeed(
        DATA_DIR / f"{symbol}_{resolution}.csv",
        symbol=symbol,
        interval_seconds=3600,
    )


def _run(symbol: str, predictor=None, *, strategy: str | None = None):
    from iot_machine_learning.domain.entities.market.replay import (
        MarketReplayEngine,
        ReplayEngineConfig,
    )

    feed = _feed(symbol, "1h")
    engine = MarketReplayEngine(
        ReplayEngineConfig(
            symbol=symbol,
            feed=feed,
            interval_seconds=3600,
            horizons_seconds=HORIZONS_1H,
            strategy=strategy or ("baseline" if predictor is None else "x"),
            predictor=predictor,
        )
    )
    return engine.run()


def _baselines_section() -> str:
    from iot_machine_learning.domain.entities.market.replay import BASELINES

    lines = ["1) BASELINES — NVDA 1h (todo el histórico)", ""]
    header = f"{'baseline':<18}{'horizon':<8}{'dir%':>8}{'brier':>8}{'reward':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for predictor in BASELINES:
        outcome = _run_by_baseline(predictor)
        for label, _, stats in outcome:
            lines.append(
                f"{predictor.name:<18}{label:<8}{stats[0]:>7.1f}%"
                f"{stats[1]:>8.3f}{stats[2]:>+10.1f}"
            )
    return "\n".join(lines)


def _run_by_baseline(predictor):
    from iot_machine_learning.domain.entities.market.replay import (
        MetricCollector,
        MetricKey,
    )

    result = _run("NVDA", predictor=predictor, strategy=predictor.name)
    collector = MetricCollector()
    for pred in result.predictions:
        if pred.outcome is not None:
            collector.add(
                MetricKey(instrument="NVDA", horizon_seconds=pred.horizon_seconds),
                pred,
                pred.outcome,
            )
    out = []
    for key in collector.keys():
        metrics = collector.metrics(key)
        label = _fmt_horizon(key.horizon_seconds)
        out.append((label, key.horizon_seconds, (metrics.direction_accuracy * 100, metrics.brier, metrics.reward)))
    return sorted(out, key=lambda row: row[1])


def _fmt_horizon(seconds: int) -> str:
    if seconds == 3600:
        return "1h"
    if seconds == 14400:
        return "4h"
    if seconds == 86400:
        return "1d"
    return f"{seconds}s"


def _walk_forward_section() -> str:
    from iot_machine_learning.domain.entities.market.replay import (
        MarketReplayEngine,
        NaivePredictor,
        RandomPredictor,
        ReplayEngineConfig,
        TrainedMomentumPredictor,
        split_walk_forward,
    )

    lines = ["2) WALK-FORWARD — NVDA 1h (train 90d -> test 15d, step 15d)", ""]
    feed = _feed("NVDA", "1h")
    candles = tuple(feed.iter_events())
    splits = split_walk_forward(
        candles,
        train_seconds=90 * 86400,
        test_seconds=15 * 86400,
        step_seconds=15 * 86400,
        min_train=350,
    )
    header = (
        f"{'window':<8}{'trained%':>9}{'naive%':>9}{'random%':>9}"
        f"{'n':>7}{'regime':<16}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for i, split in enumerate(splits, start=1):
        trained = TrainedMomentumPredictor()
        trained.fit(split.train, horizon_seconds=86400)
        outcome = []
        for label, predictor in (
            ("trained", trained),
            ("naive", NaivePredictor()),
            ("random", RandomPredictor()),
        ):
            engine = MarketReplayEngine(
                ReplayEngineConfig(
                    symbol="NVDA",
                    feed=_TupleFeed(split.test),
                    interval_seconds=3600,
                    horizons_seconds=HORIZONS_1H,
                    strategy=label,
                    predictor=predictor,
                )
            )
            result = engine.run()
            hits = sum(
                1
                for p in result.predictions
                if p.outcome is not None
                and (p.probability_up >= 0.5) == (p.outcome.return_realized > 0)
            )
            n = sum(1 for p in result.predictions if p.outcome is not None)
            outcome.append((hits / n * 100) if n else 0.0)
        regime = _segment_regime(tuple(split.test))
        lines.append(
            f"{i:<8}{outcome[0]:>8.1f}%{outcome[1]:>8.1f}%{outcome[2]:>8.1f}%"
            f"{len(split.test):>7} {regime:<16}"
        )
    return "\n".join(lines)


class _TupleFeed:
    symbol = "NVDA"

    def __init__(self, candles) -> None:
        self._candles = candles

    def iter_events(self):
        yield from self._candles


def _segment_regime(candles) -> str:
    from iot_machine_learning.domain.entities.market.replay import (
        FeatureWindow,
        classify_window,
    )

    window = FeatureWindow(symbol="NVDA", candles=tuple(candles[:REGIME_LOOKBACK]))
    if window.size < 21:
        return "—"
    return classify_window(window).value


def _matrix_section(symbols: list[str]) -> str:
    from iot_machine_learning.domain.entities.market.replay import (
        FeatureWindow,
        MarketRegime,
        MetricCollector,
        MetricKey,
        classify_window,
    )

    lines = ["3) ZENIN PERFORMANCE MATRIX — direction % por régimen x horizonte", ""]
    header = f"{'regime':<18}" + "".join(f"{_fmt_horizon(h):>8}" for h in HORIZONS_1H)
    lines.append(header)
    lines.append("-" * len(header))

    segment_counts: dict[str, int] = {}
    m = MetricCollector()
    symbol_counts: dict[str, int] = {}
    for symbol in symbols:
        candles = tuple(_feed(symbol, "1h").iter_events())
        result = _run(symbol)
        segments: list[tuple[str, float, float]] = []
        for start in range(0, len(candles) - REGIME_LOOKBACK, SEGMENT_VELAS):
            segment = tuple(candles[start : start + SEGMENT_VELAS])
            if len(segment) < REGIME_LOOKBACK:
                break
            window = FeatureWindow(symbol=symbol, candles=tuple(segment[:REGIME_LOOKBACK]))
            regime = classify_window(window)
            segments.append((regime.value, segment[0].timestamp, segment[-1].timestamp))
            segment_counts[regime.value] = segment_counts.get(regime.value, 0) + 1
        for pred in result.predictions:
            if pred.outcome is None:
                continue
            ts = pred.observation.timestamp
            regime_str: str | None = None
            for name, ts_lo, ts_hi in segments:
                if ts_lo <= ts < ts_hi:
                    regime_str = name
                    break
            if regime_str is None:
                continue
            m.add(
                MetricKey(instrument=symbol, horizon_seconds=pred.horizon_seconds, regime=regime_str),
                pred,
                pred.outcome,
            )
        symbol_counts[symbol] = len(result.predictions)

    for regime in MarketRegime:
        row = f"{regime.value:<18}"
        for horizon in HORIZONS_1H:
            total_n = 0
            total_hits = 0.0
            for key in m.keys():
                if key.horizon_seconds != horizon or key.regime != regime.value:
                    continue
                metrics = m.metrics(key)
                total_n += metrics.n
                total_hits += metrics.direction_accuracy * metrics.n
            if total_n == 0:
                row += f"{'—':>8}"
            else:
                row += f"{total_hits / total_n * 100:>7.0f}%"
        lines.append(row)
    lines.append("")
    segments_txt = ", ".join(
        f"{k}={v}" for k, v in sorted(segment_counts.items())
    )
    symbols_txt = ", ".join(f"{k}:{v}" for k, v in symbol_counts.items())
    lines.append(f"Segmentos por régimen: {segments_txt}")
    lines.append(f"Predicciones por símbolo: {symbols_txt}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(_baselines_section())
    print()
    print(_walk_forward_section())
    print()
    print(_matrix_section(symbols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
