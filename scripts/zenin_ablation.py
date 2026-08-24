#!/usr/bin/env python
"""ZENIN MARKET — ABLATION MATRIX (FASE 9.3).

La pregunta: ¿de dónde salió el edge bruto de FASE 9.2? No se agrega
nada a ZENIN: se REMUEVE un componente a la vez y se mide sobre los
MISMOS outcomes de las corridas reales (predicciones persistidas en el
store). El engine NO se re-corre: solo cambia el vector de pesos
aplicado a los expertos por ventana.

    Naive / Momentum / EMA crossover   — el experto solo (peso 1.0)
    ZENIN - memoria                    — pesos uniformes (sin adaptación)
    ZENIN - régimen                    — contexto global sin dimensión régimen
    ZENIN - MoE                        — contexto + mejor experto único
    ZENIN completo                     — la versión activa real

La versión activa de cada ventana se reconstruye desde la cadena
append-only de model_versions (reason "wf {symbol} W{index}"): la última
creada por ese símbolo con índice ≤ W, o la heredada al iniciar la
corrida (contaminación inter-símbolo incluida — es lo que ocurrió).

Uso:
    python scripts/zenin_ablation.py
    python scripts/zenin_ablation.py --symbols NVDA,AMD,AAPL,BTC-USD
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

_ST_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV, override=True)

from iot_machine_learning.domain.entities.market.adaptation import (  # noqa: E402
    PerformanceAnalyzer,
)
from iot_machine_learning.domain.entities.market.costs import COST_PROFILES  # noqa: E402
from iot_machine_learning.domain.entities.market.replay import (  # noqa: E402
    wf_windows,
    window_regime,
)
from iot_machine_learning.domain.entities.market.replay.ablation import (  # noqa: E402
    ABLATIONS,
    AblationStats,
    AblationWindow,
    ablation_weights,
    ablation_window_stats,
    active_versions_by_window,
    aggregate_ablation,
    portfolio_net_returns,
    render_ablation_matrix,
)
from iot_machine_learning.infrastructure.adapters.market import (  # noqa: E402
    RESOLUTIONS,
    HistoricalCsvFeed,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market import (  # noqa: E402
    ZeninMarketDbConnection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="NVDA,AMD,AAPL,BTC-USD")
    parser.add_argument("--interval", choices=list(RESOLUTIONS), default="1h")
    parser.add_argument("--train-days", type=float, default=14.0)
    parser.add_argument("--test-days", type=float, default=7.0)
    parser.add_argument(
        "--step-days", type=float, default=None, help="avance del origin (default: test-days)"
    )
    args = parser.parse_args()

    interval, default_horizons = RESOLUTIONS[args.interval]
    horizons = default_horizons
    step_seconds = args.step_days * 86400 if args.step_days is not None else args.test_days * 86400
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    if not ZeninMarketDbConnection.health_check():
        print("MySQL zenin_market no disponible: revisa .env (MYSQL_*)")
        return 1

    windows_by_symbol: dict[str, list] = {}
    for symbol in symbols:
        path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "market"
            / f"{symbol}_{args.interval}.csv"
        )
        if not path.exists():
            print(f"dataset no existe: {path}")
            return 1
        feed = HistoricalCsvFeed(path, symbol=symbol, interval_seconds=interval)
        candles = tuple(feed.iter_events())
        if not candles:
            print(f"dataset vacío: {path}")
            return 1
        windows_by_symbol[symbol] = list(
            wf_windows(
                candles,
                train_seconds=args.train_days * 86400,
                test_seconds=args.test_days * 86400,
                step_seconds=step_seconds,
            )
        )

    analyzer = PerformanceAnalyzer(calibration_penalty=0.5)
    window_rows: list[AblationWindow] = []
    pooled: dict[tuple[str, str], list[float]] = {}
    window_nets: dict[tuple[str, str], list[float]] = {}
    pooled_regime: dict[tuple[str, str, str], list[float]] = {}
    window_nets_regime: dict[tuple[str, str, str], list[float]] = {}
    cost_bps_map: dict[str, int] = {}
    window_counts: dict[str, int] = {}

    with ZeninMarketDbConnection.get_connection() as conn:
        version_rows = [
            dict(row)
            for row in conn.execute(
                text(
                    "SELECT version_id, created_at, weights, reason "
                    "FROM model_versions ORDER BY version_id"
                )
            ).mappings().all()
        ]
        for symbol in symbols:
            windows = windows_by_symbol[symbol]
            window_counts[symbol] = len(windows)
            cost_bps_map[symbol] = COST_PROFILES.get(symbol, COST_PROFILES["NVDA"]).total_bps
            cost = cost_bps_map[symbol]
            active = active_versions_by_window(
                version_rows, symbol, [w.index for w in windows]
            )

            for window in windows:
                w = window
                version = active.get(w.index)
                if version is None or not version.get("weights"):
                    weights_by_context: dict[str, dict[str, float]] = {}
                else:
                    weights_by_context = {
                        str(ctx): dict(weights)
                        for ctx, weights in json.loads(version["weights"]).items()
                    }
                regime = window_regime(w.train)
                scores = analyzer.analyze(
                    conn.execute(
                        text(
                            "SELECT strategy, regime, horizon_seconds, "
                            "COUNT(*) AS evaluated, "
                            "SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) AS hits, "
                            "AVG(calibration_error) AS calibration, "
                            "AVG(expected_return) AS expected_return, "
                            "AVG(outcome_return_realized) AS realized_return, "
                            "COALESCE(SUM(reward_total), 0.0) AS reward "
                            "FROM market_predictions "
                            "WHERE symbol = :symbol AND status = 'rewarded' "
                            "AND (data_status IS NULL OR data_status <> 'stale') "
                            "AND emitted_at >= :since AND emitted_at < :until "
                            "GROUP BY strategy, regime, horizon_seconds "
                            "ORDER BY strategy, regime, horizon_seconds"
                        ),
                        {
                            "symbol": symbol,
                            "since": w.test_start,
                            "until": w.test_end,
                        },
                    )
                    .mappings()
                    .all()
                )
                per_ts: dict[float, dict[str, tuple[bool, float]]] = {}
                for row in conn.execute(
                    text(
                        "SELECT strategy, emitted_at, direction_correct, "
                        "outcome_return_realized "
                        "FROM market_predictions "
                        "WHERE symbol = :symbol AND status = 'rewarded' "
                        "AND emitted_at >= :since AND emitted_at < :until "
                        "ORDER BY emitted_at"
                    ),
                    {
                        "symbol": symbol,
                        "since": w.test_start,
                        "until": w.test_end,
                    },
                ).mappings().all():
                    if row["outcome_return_realized"] is None:
                        continue
                    per_ts.setdefault(float(row["emitted_at"]), {})[
                        str(row["strategy"])
                    ] = (bool(row["direction_correct"]), float(row["outcome_return_realized"]))
                per_timestamp = sorted(per_ts.items())

                for ablation in ABLATIONS:
                    weights = ablation_weights(
                        ablation,
                        weights_by_context=weights_by_context,
                        regime=regime,
                        horizon=horizons[0],
                        scores=scores,
                    )
                    if weights is None:
                        continue
                    scoped = [
                        s for s in scores if s.regime == regime
                    ] if regime is not None else list(scores)
                    expected = {s.expert: s.expected_return for s in scoped}
                    accuracy = {s.expert: s.accuracy for s in scoped}
                    stats = ablation_window_stats(
                        symbol=symbol,
                        index=w.index,
                        regime=regime,
                        ablation=ablation,
                        cost_bps=cost,
                        weights=weights,
                        expected=expected,
                        accuracy=accuracy,
                        per_timestamp=per_timestamp,
                    )
                    window_rows.append(stats)
                    net_series = portfolio_net_returns(
                        weights, per_timestamp, cost / 10000.0
                    )
                    pooled.setdefault((symbol, ablation), []).extend(net_series)
                    window_nets.setdefault((symbol, ablation), []).append(
                        stats.realized_net
                    )
                    pooled_regime.setdefault((symbol, ablation, regime), []).extend(
                        net_series
                    )
                    window_nets_regime.setdefault((symbol, ablation, regime), []).append(
                        stats.realized_net
                    )

    stats_by_symbol: dict[str, dict[str, AblationStats]] = {}
    by_regime: dict[str, dict[tuple[str, str], AblationStats]] = {}
    for symbol in symbols:
        stats_by_symbol[symbol] = {
            ablation: aggregate_ablation(
                [w for w in window_rows if w.symbol == symbol and w.ablation == ablation],
                pooled.get((symbol, ablation), []),
                window_nets.get((symbol, ablation), []),
            )
            for ablation in ABLATIONS
        }
        for (sym, ablation, regime), returns in pooled_regime.items():
            if sym != symbol:
                continue
            by_regime.setdefault(symbol, {})[(ablation, regime)] = aggregate_ablation(
                [
                    w
                    for w in window_rows
                    if w.symbol == symbol and w.ablation == ablation and w.regime == regime
                ],
                returns,
                window_nets_regime.get((sym, ablation, regime), []),
            )

    print(
        render_ablation_matrix(
            stats_by_symbol,
            by_regime,
            cost_bps=cost_bps_map,
            window_counts=window_counts,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
