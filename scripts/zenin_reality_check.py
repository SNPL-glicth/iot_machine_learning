#!/usr/bin/env python
"""ZENIN MARKET — STATISTICAL REALITY CHECK (FASE 9.5).

¿Es la señal estadísticamente real, o ruido/estructura del dataset?

Todo se computa sobre los outcomes persistidos de las corridas 9.1/9.2
(store; el engine NO se re-corre) y con selección que usa SOLO
información TRAIN (sin lookahead). Pruebas, por símbolo:

    1. PERMUTACIÓN TEMPORAL  — predicciones intactas, outcomes
       barajados entre timestamps. Bajo el nulo E[PnL] = 0 antes de
       costos. Si el edge real sobrevive a la permutación → es
       estructura del dataset, no capacidad predictiva.

    2. BOOTSTRAP POR VENTANA  — IC 95% percentil (2000 remuestras)
       para accuracy, net, sharpe y maxDD del agregado real.

    3. DIFERENCIA VS BASELINE  — ZENIN − (Naive | EMA) por ventana
       con IC 95%. Si el IC cruza cero no hay superioridad demostrable.

    4. PERMUTACIÓN DEL GANADOR  — destruir solo la asociación
       contexto → experto ganador (ganador aleatorio por ventana).
       Responde si la SELECCIÓN aporta o es ruido (FASE 9.3).

    5. BOOTSTRAP POR EXPERTO  — IC 95% para accuracy, mean_reward
       y ECE de cada estrategia.

Opcional (--val-days): walk-forward agresivo entrenamiento →
validación → test. La validación elige el modo de selección por
ventana (soft/selective/hard_max por edge neto de validación); el
TEST jamás se usa para decidir.

Uso:
    python scripts/zenin_reality_check.py
    python scripts/zenin_reality_check.py --symbols NVDA --val-days 7 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence as _Sequence
from pathlib import Path
from typing import cast

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
from iot_machine_learning.domain.entities.market.costs import COST_PROFILES, CostModel  # noqa: E402
from iot_machine_learning.domain.entities.market.replay import (  # noqa: E402
    wf_windows,
    window_regime,
)
from iot_machine_learning.domain.entities.market.replay.ablation import (  # noqa: E402
    ABLATION_EMA,
    ABLATION_NAIVE,
    ExpertScoreLike,
    ablation_weights,
    ablation_window_stats,
    portfolio_net_returns,
)
from iot_machine_learning.domain.entities.market.replay.significance import (  # noqa: E402
    PermWindow,
    WindowRecord,
    block_bootstrap,
    bootstrap_expert_metrics,
    difference_ci,
    permutation_test,
    pooled_sharpe,
    random_winner_test,
    weighted_acc,
    weighted_net,
    window_cumsum_maxdd,
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


def _per_timestamp(
    conn, symbol: str, since: float, until: float
) -> list[tuple[float, dict[str, tuple[bool, float]]]]:
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
        {"symbol": symbol, "since": since, "until": until},
    ).mappings().all():
        if row["outcome_return_realized"] is None:
            continue
        per_ts.setdefault(float(row["emitted_at"]), {})[str(row["strategy"])] = (
            bool(row["direction_correct"]),
            float(row["outcome_return_realized"]),
        )
    return sorted(per_ts.items())


def _fmt_ci(ci, *, kind: str = "pct") -> str:
    if kind == "num":
        return f"[{ci.ci_low:+.2f}, {ci.ci_high:+.2f}]"
    return f"[{ci.ci_low:+.2%}, {ci.ci_high:+.2%}]"


def render_report(
    *,
    symbol_stats: dict,
    seed: int,
    n_permutations: int,
    n_boot: int,
    n_random_winner: int,
    val_days: float,
) -> str:
    out: list[str] = []
    out.append("STATISTICAL REALITY CHECK — ZENIN MARKET (FASE 9.5)")
    out.append(f"seed={seed} · permutaciones={n_permutations} · bootstrap={n_boot} · "
               f"ganador aleatorio={n_random_winner} · validación={val_days} días")
    out.append("")
    for symbol, data in symbol_stats.items():
        out.append(
            f"== {symbol} ({data['cost_bps']} bps, {data['window_count']} ventanas, "
            f"{data['hold_count']} hold) =="
        )
        out.append("")
        # 1) permutación temporal
        out.append(f"permutación temporal — predicciones intactas, outcomes barajados "
                   f"({n_permutations})")
        header = f"{'modo':<10}{'real':>10}{'nulo±std':>14}{'IC95% nulo':>24}{'p':>8}"
        out.append(header)
        for mode, result in data["permutations"].items():
            out.append(
                f"{mode:<10}{result.real_mean:>+10.2%}"
                f"{result.null_mean:>+8.2%}±{result.null_std:.2%}"
                f"{_fmt_ci(result):>24}{result.p_value:>8.3f}"
            )
        if data.get("permutation_chosen") is not None:
            r = data["permutation_chosen"]
            out.append(
                f"{'chosen':<10}{r.real_mean:>+10.2%}"
                f"{r.null_mean:>+8.2%}±{r.null_std:.2%}"
                f"{_fmt_ci(r):>24}{r.p_value:>8.3f}"
            )
        out.append("")
        # 2) bootstrap por ventana
        out.append(f"bootstrap por ventana ({n_boot}) — IC 95% percentil")
        out.append(
            f"{'modo':<10}{'n':>6}{'acc':>7}{'IC acc':>18}{'net':>9}{'IC net':>18}"
            f"{'sharpe':>8}{'IC sharpe':>18}{'maxDD':>8}{'IC maxDD':>18}"
        )
        for mode, stats in data["bootstrap"].items():
            acc, net, sharpe, dd = stats
            out.append(
                f"{mode:<10}{data['n'].get(mode, 0):>6}"
                f"{acc.point:>7.1%}{_fmt_ci(acc):>18}"
                f"{net.point:>+9.2%}{_fmt_ci(net):>18}"
                f"{sharpe.point:>+8.2f}{_fmt_ci(sharpe, kind='num'):>18}"
                f"{dd.point:>+8.2%}{_fmt_ci(dd):>18}"
            )
        out.append("")
        # 3) diferencia vs baselines
        out.append("diferencia vs baseline (ZENIN hard_max − baseline, IC 95%)")
        out.append(f"{'baseline':<10}{'diff':>10}{'IC':>24}{'cruza cero':>12}")
        for baseline, ci in data["baseline_diff"].items():
            out.append(
                f"{baseline:<10}{ci.point:>+10.2%}{_fmt_ci(ci):>24}"
                f"{'SÍ' if ci.crosses_zero else 'NO':>12}"
            )
        out.append("")
        # 4) permutación del ganador
        rw = data["random_winner"]
        out.append(f"ganador aleatorio ({n_random_winner}) — ¿la selección importa?")
        out.append(
            f"{'real':>10}{'nulo±std':>14}{'IC95% nulo':>24}{'p':>8}\n"
            f"{rw.real_mean:>+10.2%}{rw.null_mean:>+8.2%}±{rw.null_std:.2%}"
            f"{_fmt_ci(rw):>24}{rw.p_value:>8.3f}"
        )
        out.append("")
        # 5) expertos
        out.append(f"expertos — bootstrap ({n_boot // 2}), máx {data['expert_max_rows']} filas")
        out.append(
            f"{'estrategia':<14}{'n':>6}{'acc':>7}{'IC acc':>18}{'reward':>9}"
            f"{'IC reward':>18}{'ECE':>7}{'IC ECE':>18}"
        )
        for expert, metrics in data["expert_metrics"].items():
            out.append(
                f"{expert:<14}{metrics.n:>6}"
                f"{metrics.accuracy.point:>7.1%}{_fmt_ci(metrics.accuracy):>18}"
                f"{metrics.mean_reward.point:>+9.2f}{_fmt_ci(metrics.mean_reward, kind='num'):>18}"
                f"{metrics.ece.point:>7.1%}{_fmt_ci(metrics.ece):>18}"
            )
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="NVDA,AMD,AAPL,BTC-USD")
    parser.add_argument("--interval", choices=list(RESOLUTIONS), default="1h")
    parser.add_argument("--train-days", type=float, default=14.0)
    parser.add_argument("--test-days", type=float, default=7.0)
    parser.add_argument("--val-days", type=float, default=0.0,
                        help="walk-forward agresivo: validación entre train y test")
    parser.add_argument("--step-days", type=float, default=None)
    parser.add_argument("--risk-aversion", type=float, default=0.1)
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--max-experts", type=int, default=2)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-random-winner", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert-max-rows", type=int, default=3000)
    args = parser.parse_args()

    interval, _ = RESOLUTIONS[args.interval]
    step_seconds = args.step_days * 86400 if args.step_days is not None else args.test_days * 86400
    train_seconds = (args.train_days + args.val_days) * 86400
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
                train_seconds=train_seconds,
                test_seconds=args.test_days * 86400,
                step_seconds=step_seconds,
            )
        )

    analyzer = PerformanceAnalyzer(calibration_penalty=0.5)
    symbol_stats: dict = {}

    with ZeninMarketDbConnection.get_connection() as conn:
        repo = MarketPredictionRepository(conn)
        for symbol in symbols:
            windows = windows_by_symbol[symbol]
            cost_bps = int(
                COST_PROFILES.get(symbol, COST_PROFILES["NVDA"]).total_bps
            )
            profile = COST_PROFILES.get(symbol, COST_PROFILES["NVDA"])
            cost_model = CostModel(
                spread_bps=profile.spread_bps,
                slippage_bps=profile.slippage_bps,
                commission_bps=profile.commission_bps,
            )
            cost = cost_bps / 10000.0

            stats_by_mode: dict[str, list] = {m.value: [] for m in MODES}
            pooled: dict[str, list[float]] = {m.value: [] for m in MODES}
            records: dict[str, list[WindowRecord]] = {m.value: [] for m in MODES}
            perm_windows: dict[str, list[PermWindow]] = {m.value: [] for m in MODES}
            hold_count = 0
            hard_nets_by_window: dict[int, float] = {}
            baseline_pairs: dict[str, list[tuple[float, float]]] = {"naive": [], "ema": []}
            baseline_pair_weights: dict[str, list[int]] = {"naive": [], "ema": []}
            chosen_stats: list = []
            chosen_perm: list[PermWindow] = []
            chosen_mode_counts: dict[str, int] = {}

            for window in windows:
                w = window
                regime = window_regime(w.train)
                train_end = w.train_start + args.train_days * 86400
                train_rows = repo.expert_performance(
                    symbol=symbol, since=w.train_start, until=train_end
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
                test_per_ts = _per_timestamp(conn, symbol, w.test_start, w.test_end)
                expected = {s.expert: s.expected_return for s in scoped}
                accuracy = {s.expert: s.accuracy for s in scoped}

                mode_weights: dict[str, dict[str, float]] = {}
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
                        continue
                    mode_weights[mode.value] = result.weights

                # ── validación: elige el modo por edge neto de validación ──
                chosen: str | None = None
                if args.val_days > 0 and mode_weights:
                    val_per_ts = _per_timestamp(
                        conn, symbol, train_end, w.train_end
                    )
                    best_val = -float("inf")
                    for mode_name, val_weights in mode_weights.items():
                        val_nets = portfolio_net_returns(val_weights, val_per_ts, cost)
                        if not val_nets:
                            continue
                        val_mean = sum(val_nets) / len(val_nets)
                        if val_mean > best_val:
                            best_val = val_mean
                            chosen = mode_name
                if chosen is None and len(mode_weights) == 1:
                    chosen = next(iter(mode_weights))
                if chosen is not None:
                    chosen_mode_counts[chosen] = chosen_mode_counts.get(chosen, 0) + 1

                for mode in MODES:
                    mode_weights_map = mode_weights.get(mode.value)
                    if mode_weights_map is None:
                        if mode == SelectionMode.HARD_MAX:
                            hold_count += 1
                        continue
                    stats = ablation_window_stats(
                        symbol=symbol,
                        index=w.index,
                        regime=regime,
                        ablation=mode.value,
                        cost_bps=cost_bps,
                        weights=mode_weights_map,
                        expected=expected,
                        accuracy=accuracy,
                        per_timestamp=test_per_ts,
                    )
                    if stats.n == 0:
                        continue
                    stats_by_mode[mode.value].append(stats)
                    if mode.value == "hard_max":
                        hard_nets_by_window[w.index] = stats.realized_net
                    series = portfolio_net_returns(mode_weights_map, test_per_ts, cost)
                    pooled[mode.value].extend(series)
                    records[mode.value].append(
                        WindowRecord(
                            n=stats.n,
                            accuracy=stats.accuracy,
                            net=stats.realized_net,
                            returns=tuple(series),
                        )
                    )
                    perm_windows[mode.value].append(
                        PermWindow(
                            weights=mode_weights_map,
                            per_timestamp=test_per_ts,
                            cost=cost,
                            n=stats.n,
                        )
                    )
                    if mode.value == chosen:
                        chosen_stats.append(stats)
                        chosen_perm.append(
                            PermWindow(
                                weights=mode_weights_map,
                                per_timestamp=test_per_ts,
                                cost=cost,
                                n=stats.n,
                            )
                        )

                # ── baselines por ventana (Naive, EMA) ──
                for baseline, key in ((ABLATION_NAIVE, "naive"), (ABLATION_EMA, "ema")):
                    base_weights = ablation_weights(
                        baseline,
                        weights_by_context={},
                        regime=regime,
                        horizon=0,
                        scores=cast(_Sequence[ExpertScoreLike], scoped),
                    )
                    if base_weights is None:
                        continue
                    base_stats = ablation_window_stats(
                        symbol=symbol,
                        index=w.index,
                        regime=regime,
                        ablation=key,
                        cost_bps=cost_bps,
                        weights=base_weights,
                        expected=expected,
                        accuracy=accuracy,
                        per_timestamp=test_per_ts,
                    )
                    if base_stats.n == 0:
                        continue
                    hard_net = hard_nets_by_window.get(w.index)
                    if hard_net is not None:
                        baseline_pairs[key].append((hard_net, base_stats.realized_net))
                        baseline_pair_weights[key].append(base_stats.n)

            # ── pruebas de significancia por modo ──
            permutations: dict[str, object] = {}
            bootstrap: dict[str, list[object]] = {}
            for mode in MODES:
                key = mode.value
                if perm_windows[key]:
                    permutations[key] = permutation_test(
                        perm_windows[key],
                        n_permutations=args.n_permutations,
                        seed=args.seed,
                    )
                if records[key]:
                    bootstrap[key] = [
                        block_bootstrap(
                            records[key], statistic=stat, n_boot=args.n_boot, seed=args.seed
                        )
                        for stat in (weighted_acc, weighted_net, pooled_sharpe, window_cumsum_maxdd)
                    ]
            permutation_chosen = None
            if chosen_perm:
                permutation_chosen = permutation_test(
                    chosen_perm,
                    n_permutations=args.n_permutations,
                    seed=args.seed,
                )

            baseline_diff: dict[str, object] = {}
            for key in ("naive", "ema"):
                if baseline_pairs[key]:
                    baseline_diff[key] = difference_ci(
                        baseline_pairs[key],
                        baseline_pair_weights[key],
                        n_boot=args.n_boot,
                        seed=args.seed,
                    )

            random_winner = None
            if perm_windows["hard_max"]:
                random_winner = random_winner_test(
                    perm_windows["hard_max"],
                    n_permutations=args.n_random_winner,
                    seed=args.seed,
                )

            expert_rows: dict[str, list[tuple[bool, float, float]]] = {}
            for row in conn.execute(
                text(
                    "SELECT strategy, direction_correct, reward_total, "
                    "calibration_error "
                    "FROM market_predictions "
                    "WHERE symbol = :symbol AND status = 'rewarded' "
                    "AND (data_status IS NULL OR data_status <> 'stale') "
                    "AND outcome_return_realized IS NOT NULL "
                    "ORDER BY strategy"
                ),
                {"symbol": symbol},
            ).mappings().all():
                expert_rows.setdefault(str(row["strategy"]), []).append(
                    (bool(row["direction_correct"]),
                     float(row["reward_total"] or 0.0),
                     float(row["calibration_error"] or 0.0))
                )
            expert_metrics = {
                expert: bootstrap_expert_metrics(
                    rows, n_boot=args.n_boot // 2, seed=args.seed,
                    max_rows=args.expert_max_rows,
                )
                for expert, rows in expert_rows.items()
            }

            symbol_stats[symbol] = {
                "cost_bps": cost_bps,
                "window_count": len(windows),
                "hold_count": hold_count,
                "permutations": permutations,
                "permutation_chosen": permutation_chosen,
                "bootstrap": bootstrap,
                "baseline_diff": baseline_diff,
                "random_winner": random_winner,
                "expert_metrics": expert_metrics,
                "expert_max_rows": args.expert_max_rows,
                "n": {m.value: sum(r.n for r in records[m.value]) for m in MODES},
            }

    print(
        render_report(
            symbol_stats=symbol_stats,
            seed=args.seed,
            n_permutations=args.n_permutations,
            n_boot=args.n_boot,
            n_random_winner=args.n_random_winner,
            val_days=args.val_days,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
