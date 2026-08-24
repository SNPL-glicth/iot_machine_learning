#!/usr/bin/env python
"""ZENIN MARKET — ADAPTIVE EXPERT SELECTION (FASE 9.4).

FASE 9.3 demostró que la señal de ZENIN vive en la SELECCIÓN del mejor
experto (MoE hard-max: 68.4% NVDA, 60.0% AMD, 62.1% AAPL, 54.1% BTC),
pero que argmax → operar a ciegas amplifica el ruido de selección.
9.4 compara tres modos de selección controlada sobre los MISMOS
outcomes de las corridas reales (store; el engine NO se re-corre):

    soft       — softmax sobre el score neto (todo el contexto pesa)
    selective  — solo los mejores expertos con peso significativo
    hard_max   — ganador único, SOLO si pasa guardrails (n, historial,
                 margen sobre el segundo, edge neto > 0)

La decisión de selección usa EXCLUSIVAMENTE información TRAIN (sin
lookahead: al inicio de cada ventana el selector solo conoce lo que
ocurrió hasta train_end). El score deja de ser accuracy (métrica
secundaria) y pasa a:

    score = expected_net × calibration_quality × evidence_strength
    expected_net = expected_return − costo − riesgo_aversion × PnL_std

Y la gran novedad: la puerta NO TRADE. Si el mejor experto del
contexto no tiene edge neto esperado > 0, la ventana declara HOLD:
ZENIN no está obligado a escoger.

Uso:
    python scripts/zenin_selection.py
    python scripts/zenin_selection.py --symbols NVDA,AMD,AAPL,BTC-USD
"""

from __future__ import annotations

import argparse
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
from iot_machine_learning.domain.entities.market.adaptation.selection import (  # noqa: E402
    SelectionConfig,
    SelectionMode,
    expert_net_scores,
    select_weights,
)
from iot_machine_learning.domain.entities.market.costs import (  # noqa: E402
    COST_PROFILES,
    CostModel,
)
from iot_machine_learning.domain.entities.market.replay import (  # noqa: E402
    wf_windows,
    window_regime,
)
from iot_machine_learning.domain.entities.market.replay.ablation import (  # noqa: E402
    AblationStats,
    AblationWindow,
    ablation_window_stats,
    aggregate_ablation,
    portfolio_net_returns,
)
from iot_machine_learning.infrastructure.adapters.market import (  # noqa: E402
    RESOLUTIONS,
    HistoricalCsvFeed,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market import (  # noqa: E402
    ZeninMarketDbConnection,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.market_prediction_repository import (  # noqa: E402
    MarketPredictionRepository,
)

MODES: tuple[SelectionMode, ...] = (
    SelectionMode.SOFT,
    SelectionMode.SELECTIVE,
    SelectionMode.HARD_MAX,
)


def render_report(
    stats_by_symbol: dict[str, dict[str, AblationStats]],
    *,
    cost_bps: dict[str, int],
    window_counts: dict[str, int],
    holds: dict[tuple[str, str], int],
    risk_aversion: float,
) -> str:
    out: list[str] = []
    out.append("ADAPTIVE EXPERT SELECTION — ZENIN MARKET (FASE 9.4)")
    out.append(
        f"score = expected_net × calibración × evidencia · "
        f"riesgo_aversion={risk_aversion} · decisión solo con TRAIN"
    )
    out.append("")
    header = (
        f"{'modo':<10}{'ventanas':>9}{'hold':>6}{'n':>7}{'acc':>7}"
        f"{'exp':>9}{'net':>9}{'sharpe':>8}{'maxDD':>9}"
    )
    for symbol in stats_by_symbol:
        out.append(
            f"== {symbol} ({cost_bps.get(symbol, 0)} bps, "
            f"{window_counts.get(symbol, 0)} ventanas) =="
        )
        out.append(header)
        for mode in MODES:
            stats = stats_by_symbol[symbol][mode.value]
            if stats is None:
                continue
            wins = window_counts.get(symbol, 0)
            held = holds.get((symbol, mode.value), 0)
            if stats.n == 0:
                out.append(
                    f"{mode.value:<10}{wins:>9}{held:>6}{'-':>7}{'-':>7}"
                    f"{'-':>9}{'-':>9}{'-':>9}{'-':>8}{'-':>9}"
                )
                continue
            out.append(
                f"{mode.value:<10}{wins:>9}{held:>6}"
                f"{stats.n:>7}{stats.accuracy:>7.1%}"
                f"{stats.gross_edge:>+9.2%}{stats.net_edge:>+9.2%}"
                f"{stats.sharpe:>+8.2f}{stats.max_drawdown:>+9.2%}"
            )
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="NVDA,AMD,AAPL,BTC-USD")
    parser.add_argument("--interval", choices=list(RESOLUTIONS), default="1h")
    parser.add_argument("--train-days", type=float, default=14.0)
    parser.add_argument("--test-days", type=float, default=7.0)
    parser.add_argument(
        "--step-days", type=float, default=None, help="avance del origin (default: test-days)"
    )
    parser.add_argument("--risk-aversion", type=float, default=0.1)
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--max-experts", type=int, default=2)
    args = parser.parse_args()

    interval, default_horizons = RESOLUTIONS[args.interval]
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
    holds: dict[tuple[str, str], int] = {}
    cost_bps_map: dict[str, int] = {}
    window_counts: dict[str, int] = {}

    with ZeninMarketDbConnection.get_connection() as conn:
        repo = MarketPredictionRepository(conn)
        for symbol in symbols:
            windows = windows_by_symbol[symbol]
            window_counts[symbol] = len(windows)
            cost_bps_map[symbol] = int(
                COST_PROFILES.get(symbol, COST_PROFILES["NVDA"]).total_bps
            )
            cost_model = CostModel(
                spread_bps=COST_PROFILES.get(symbol, COST_PROFILES["NVDA"]).spread_bps,
                slippage_bps=COST_PROFILES.get(symbol, COST_PROFILES["NVDA"]).slippage_bps,
                commission_bps=COST_PROFILES.get(symbol, COST_PROFILES["NVDA"]).commission_bps,
            )

            for window in windows:
                w = window
                regime = window_regime(w.train)
                # ── selección con SOLO información TRAIN (sin lookahead) ──
                train_rows = repo.expert_performance(
                    symbol=symbol, since=w.train_start, until=w.train_end
                )
                train_scores = analyzer.analyze(train_rows)
                scoped = (
                    [s for s in train_scores if s.regime == regime]
                    if regime is not None
                    else list(train_scores)
                )
                net_scores = expert_net_scores(
                    scoped,
                    cost_model=cost_model,
                    risk_aversion=args.risk_aversion,
                    min_n=args.min_n,
                )

                # ── outcomes reales del span TEST (persistidos) ──
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

                expected = {s.expert: s.expected_return for s in scoped}
                accuracy = {s.expert: s.accuracy for s in scoped}
                cost = cost_bps_map[symbol]

                for mode in MODES:
                    result = select_weights(
                        net_scores,
                        config=SelectionConfig(
                            mode=mode,
                            risk_aversion=args.risk_aversion,
                            min_n=args.min_n,
                            min_margin=args.min_margin,
                            max_experts=args.max_experts,
                        ),
                    )
                    if result.decision == "hold" or not result.weights:
                        holds[(symbol, mode.value)] = holds.get((symbol, mode.value), 0) + 1
                        continue
                    stats = ablation_window_stats(
                        symbol=symbol,
                        index=w.index,
                        regime=regime,
                        ablation=mode.value,
                        cost_bps=cost,
                        weights=result.weights,
                        expected=expected,
                        accuracy=accuracy,
                        per_timestamp=per_timestamp,
                    )
                    window_rows.append(stats)
                    net_series = portfolio_net_returns(
                        result.weights, per_timestamp, cost / 10000.0
                    )
                    pooled.setdefault((symbol, mode.value), []).extend(net_series)
                    window_nets.setdefault((symbol, mode.value), []).append(stats.realized_net)

    stats_by_symbol: dict[str, dict[str, AblationStats]] = {}
    for symbol in symbols:
        stats_by_symbol[symbol] = {
            mode.value: aggregate_ablation(
                [
                    w
                    for w in window_rows
                    if w.symbol == symbol and w.ablation == mode.value
                ],
                pooled.get((symbol, mode.value), []),
                window_nets.get((symbol, mode.value), []),
            )
            for mode in MODES
        }

    print(
        render_report(
            stats_by_symbol,
            cost_bps=cost_bps_map,
            window_counts=window_counts,
            holds=holds,
            risk_aversion=args.risk_aversion,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
