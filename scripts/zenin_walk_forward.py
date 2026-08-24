#!/usr/bin/env python
"""ZENIN MARKET — WALK-FORWARD VALIDATION (FASE 9.1).

La pregunta no es "¿ajusta bien?" sino "¿sobrevive fuera de muestra?".

    TRAIN       TEST
    ███████████ │ ██
                │
       TRAIN       TEST
       ███████████ │ ██

Por cada ventana (origin rolling):
  TRAIN: los expertos corren su shadow live sobre el fragmento TRAIN,
         se persisten y resuelven SUS outcomes (solo lo conocido hasta
         ese momento, since/until acotan el historial), y la adaptación
         MoE propone → guardrail → versión del modelo (si pasa).
  TEST:  los expertos corren sobre el fragmento TEST con los pesos de
         la versión creada en su TRAIN; el modelo compuesto se evalúa
         con outcomes reales del TEST.

Nada del futuro entra jamás en el TRAIN. Un resultado honesto (p. ej.
ZENIN 51.8% vs EMA 51.4%) es un resultado válido.

Uso:
    python scripts/zenin_walk_forward.py
    python scripts/zenin_walk_forward.py --symbol AAPL --train-days 14 --test-days 7
    python scripts/zenin_walk_forward.py --symbol BTC-USD --interval 1h --step-days 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

_ST_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV, override=True)

from iot_machine_learning.domain.entities.market.adaptation import (  # noqa: E402
    AdaptationGuard,
    PerformanceAnalyzer,
    WeightProposer,
    default_weights,
)
from iot_machine_learning.domain.entities.market.costs import (  # noqa: E402
    COST_PROFILES,
    CostModel,
    classify_edge,
)
from iot_machine_learning.domain.entities.market.prediction import Prediction  # noqa: E402
from iot_machine_learning.domain.entities.market.prediction.resolver import (  # noqa: E402
    OutcomeResolver,
)
from iot_machine_learning.domain.entities.market.replay import (  # noqa: E402
    ReplayEngineConfig,
    WfRow,
    evaluate_window,
    render_wf_report,
    wf_windows,
    window_regime,
)
from iot_machine_learning.domain.entities.market.replay.baselines import (  # noqa: E402
    BASELINES,
)
from iot_machine_learning.domain.entities.market.replay.engine import (  # noqa: E402
    MarketReplayEngine,
)
from iot_machine_learning.infrastructure.adapters.market import (  # noqa: E402
    RESOLUTIONS,
    FragmentFeed,
    HistoricalCsvFeed,
    fmt_ts,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market import (  # noqa: E402
    AdaptationRepository,
    MarketPredictionRepository,
    ZeninMarketDbConnection,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.market_prediction_repository import (  # noqa: E402
    row_to_prediction,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.migrations import (  # noqa: E402, F401
    apply_migrations,
)


class CandlePriceLookup:
    """PriceLookup sobre un feed: último cierre a lo sumo en el plazo."""

    def __init__(self, feed) -> None:
        self._closes = tuple((c.timestamp, c.close) for c in feed.iter_events())

    def last_close(self, at_or_before: float) -> float | None:
        best: float | None = None
        for ts, close in self._closes:
            if ts <= at_or_before:
                best = close
            else:
                break
        return best


def _weights_snapshot(rows: tuple[dict, ...]) -> dict[str, dict[str, float]]:
    """Contextos con muestra -> pesos uniformes iniciales (sin versión)."""
    seen: dict[tuple[str | None, int], list[str]] = {}
    for row in rows:
        context = (row["regime"], row["horizon_seconds"])
        seen.setdefault(context, []).append(str(row["strategy"]))
    return {
        f"*|{regime or '-'}|{horizon}s": default_weights(expert_names)
        for (regime, horizon), expert_names in seen.items()
    }


def _run_experts(
    *,
    full,
    experts: dict[str, object],
    names: tuple[str, ...],
    symbol: str,
    interval: int,
    horizons: tuple[int, ...],
    start: float,
    end: float,
    predictor_lookback: int,
) -> tuple[Prediction, ...]:
    """Corre los expertos sobre [start, end) con la ventana pre-calentada.

    El engine exige ``predictor_lookback + 1`` velas antes de emitir; el
    fragmento arranca ``predictor_lookback * intervalo`` antes del inicio
    y termina ``horizonte + 1`` después del final, de modo que toda
    predicción del tramo [start, end) se emite y vence dentro del feed.
    Se filtran las predicciones cuya observación queda fuera del tramo.
    """
    warmup = predictor_lookback * interval
    tail = max(horizons) + interval
    all_preds: list[Prediction] = []
    for name in names:
        fragment = FragmentFeed(full, start - warmup, end + tail)
        engine = MarketReplayEngine(
            ReplayEngineConfig(
                symbol=symbol,
                feed=fragment,
                interval_seconds=interval,
                horizons_seconds=horizons,
                predictor=experts[name],
                predictor_lookback=predictor_lookback,
            )
        )
        all_preds.extend(p for p in engine.run().predictions if p.observation.timestamp >= start)
    return tuple(all_preds)


def _sharpe(values: list[float]) -> float:
    """Sharpe de los retornos netos (media / desviación muestral)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    if var <= 0.0:
        return 0.0
    return mean / (var**0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--interval", choices=list(RESOLUTIONS), default="1h")
    parser.add_argument("--train-days", type=float, default=14.0)
    parser.add_argument("--test-days", type=float, default=7.0)
    parser.add_argument(
        "--step-days", type=float, default=None, help="avance del origin (default: test-days)"
    )
    parser.add_argument(
        "--horizons",
        default=None,
        help="horizontes en segundos separados por coma (default: los de la resolución)",
    )
    parser.add_argument("--experts", default="naive,momentum,ema-crossover,mean-reversion")
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--max-change", type=float, default=0.10)
    parser.add_argument("--min-history-days", type=int, default=2)
    parser.add_argument("--calibration-penalty", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--predictor-lookback", type=int, default=20, help="velas de contexto del motor"
    )
    parser.add_argument(
        "--cost-bps",
        type=int,
        default=None,
        help="costo total por predicción en bps (default: perfil del instrumento)",
    )
    args = parser.parse_args()

    interval, default_horizons = RESOLUTIONS[args.interval]
    horizons = (
        tuple(int(h) for h in args.horizons.split(",")) if args.horizons else default_horizons
    )
    step_seconds = args.step_days * 86400 if args.step_days is not None else args.test_days * 86400
    predictor_lookback = max(1, args.predictor_lookback)

    # FASE 9.2: perfil de costos del instrumento (override con --cost-bps).
    profile = COST_PROFILES.get(args.symbol)
    if args.cost_bps is not None:
        cost_model = CostModel(spread_bps=args.cost_bps, slippage_bps=0.0, commission_bps=0.0)
    elif profile is not None:
        cost_model = profile
    else:
        print(f"sin perfil de costos para {args.symbol}; uso default de acciones")
        cost_model = CostModel()

    experts = {p.name: p for p in BASELINES}
    requested = [name.strip() for name in args.experts.split(",") if name.strip()]
    unknown = [name for name in requested if name not in experts]
    if unknown:
        print(f"experto(s) desconocido(s): {unknown} — disponibles: {sorted(experts)}")
        return 1
    requested = [name for name in requested if name != "random"]

    path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "market"
        / f"{args.symbol}_{args.interval}.csv"
    )
    if not path.exists():
        print(f"dataset no existe: {path} (ejecuta scripts/download_market_data.py)")
        return 1
    if not ZeninMarketDbConnection.health_check():
        print(
            "MySQL zenin_market no disponible: revisa .env (MYSQL_*) y el "
            "contenedor (docker-compose.yml)"
        )
        return 1

    apply_migrations()
    full = HistoricalCsvFeed(path, symbol=args.symbol, interval_seconds=interval)
    candles = tuple(full.iter_events())
    if not candles:
        print(f"dataset vacío: {path}")
        return 1

    train_seconds = args.train_days * 86400
    test_seconds = args.test_days * 86400
    windows = wf_windows(
        candles,
        train_seconds=train_seconds,
        test_seconds=test_seconds,
        step_seconds=step_seconds,
    )
    if not windows:
        print(
            "sin ventanas: dataset o parametros sin cobertura "
            f"({len(candles)} velas, train={args.train_days}d, "
            f"test={args.test_days}d)"
        )
        return 1

    print("WALK-FORWARD — ZENIN MARKET (FASE 9.1)")
    print("TRAIN aprende SOLO lo conocido hasta t; TEST nunca alimenta TRAIN.")
    print(
        f"instrumento: {args.symbol} | resolución: {args.interval} "
        f"({interval}s) | horizontes: {horizons}"
    )
    print(
        f"ventanas: {len(windows)} | train={args.train_days}d "
        f"test={args.test_days}d step={step_seconds / 86400:.1f}d | "
        f"expertos: {requested}"
    )
    regimes = {window_regime(w.train) for w in windows}
    print(f"regímenes cubiertos: {sorted(str(r) for r in regimes)}")
    print()

    analyzer = PerformanceAnalyzer(calibration_penalty=args.calibration_penalty)
    proposer = WeightProposer(
        min_n=args.min_n,
        max_change=args.max_change,
        temperature=args.temperature,
    )
    guard = AdaptationGuard(
        min_n=args.min_n,
        min_history_days=args.min_history_days,
        max_change=args.max_change,
    )
    resolver = OutcomeResolver()

    rows = []
    with ZeninMarketDbConnection.get_connection() as conn:
        repo = MarketPredictionRepository(conn)
        adaptation = AdaptationRepository(conn)
        prices = CandlePriceLookup(full)

        for window in windows:
            w = window
            print(
                f"── W{w.index:02d} train {fmt_ts(w.train_start)} → "
                f"{fmt_ts(w.train_end)} | test {fmt_ts(w.test_start)} → "
                f"{fmt_ts(w.test_end)} (UTC)"
            )

            # ── TRAIN: correr expertos, persistir, resolver, ADAPTAR ──
            train_preds = _run_experts(
                full=full,
                experts=experts,
                names=tuple(requested),
                symbol=args.symbol,
                interval=interval,
                horizons=horizons,
                start=w.train_start,
                end=w.train_end,
                predictor_lookback=predictor_lookback,
            )
            repo.save_batch(train_preds)
            pending = repo.pending_outcomes(symbol=args.symbol)
            batch = resolver.resolve((row_to_prediction(r) for r in pending), prices)
            repo.save_batch(list(batch.resolved))

            train_scores = analyzer.analyze(
                repo.expert_performance(symbol=args.symbol, since=w.train_start, until=w.train_end)
            )
            latest = adaptation.latest_version()
            if latest is not None:
                current_weights = {
                    str(ctx): dict(weights)
                    for ctx, weights in json.loads(latest["weights"]).items()
                }
                parent = str(latest["version_id"])
            else:
                current_weights = _weights_snapshot(repo.expert_performance(symbol=args.symbol))
                parent = None

            accepted: list = []
            rejected = 0
            proposals = proposer.propose(train_scores, current_weights, parent_version=parent)
            by_context: dict[tuple[str | None, int], list] = {}
            for s in train_scores:
                by_context.setdefault((s.regime, s.horizon_seconds), []).append(s)
            vectors: dict[tuple[str | None, int], dict[str, float]] = {}
            for (regime, horizon), group in by_context.items():
                vector, _ = proposer.propose_vector(regime, horizon, group, current_weights)
                vectors[(regime, horizon)] = vector

            for proposal in proposals:
                score = next(
                    s
                    for s in train_scores
                    if (s.expert, s.regime, s.horizon_seconds)
                    == (proposal.expert, proposal.regime, proposal.horizon_seconds)
                )
                verdict = guard.evaluate(
                    proposal,
                    history_days=score.history_days,
                    data_quality="clean",
                    context_weights_after=vectors[(proposal.regime, proposal.horizon_seconds)],
                )
                adaptation.record_proposal(
                    proposal,
                    verdict,
                    proposal_id=(f"wf-{args.symbol}-W{w.index}-{uuid.uuid4().hex[:6]}"),
                )
                if verdict.passed:
                    accepted.append(proposal)
                else:
                    rejected += 1

            if accepted:
                new_weights = dict(current_weights)
                for proposal in accepted:
                    key = f"*|{proposal.regime or '-'}|{proposal.horizon_seconds}s"
                    new_weights[key] = dict(vectors[(proposal.regime, proposal.horizon_seconds)])
                version_id = adaptation.create_version(
                    weights=new_weights,
                    calibration={},
                    reason=(
                        f"wf {args.symbol} W{w.index}: {len(accepted)} propuesta(s) aceptada(s)"
                    ),
                    proposal_ids=[p.context_label for p in accepted],
                    created_at=time.time(),
                    parent_version_id=int(parent) if parent else None,
                )
                print(
                    f"    adaptación: {len(accepted)} aceptada(s), "
                    f"{rejected} rechazada(s) → MODEL v{version_id}"
                )
            else:
                version_id = int(latest["version_id"]) if latest is not None else None
                print(
                    f"    adaptación: 0 aceptadas, {rejected} rechazadas "
                    f"(modelo intacto v{version_id})"
                )

            # ── TEST: correr expertos, persistir, resolver, EVALUAR ──
            test_preds = _run_experts(
                full=full,
                experts=experts,
                names=tuple(requested),
                symbol=args.symbol,
                interval=interval,
                horizons=horizons,
                start=w.test_start,
                end=w.test_end,
                predictor_lookback=predictor_lookback,
            )
            repo.save_batch(test_preds)
            pending = repo.pending_outcomes(symbol=args.symbol)
            batch = resolver.resolve((row_to_prediction(r) for r in pending), prices)
            repo.save_batch(list(batch.resolved))

            test_scores = analyzer.analyze(
                repo.expert_performance(symbol=args.symbol, since=w.test_start, until=w.test_end)
            )
            latest = adaptation.latest_version()
            weights_by_context = {
                str(ctx): dict(weights) for ctx, weights in json.loads(latest["weights"]).items()
            }
            regime_label = window_regime(w.train)
            evals = evaluate_window(
                test_scores,
                weights_by_context,
                regime=regime_label,
                cost_model=cost_model,
            )

            # FASE 9.2: sharpe por predicción del TEST (netos realizados)
            # y clasificación del edge de la ventana (regla de piedra:
            # solo outcomes reales).
            net_returns: list[float] = []
            if evals and cost_model.total_bps:
                realized_rows = (
                    conn.execute(
                        text(
                            "SELECT outcome_return_realized FROM market_predictions "
                            "WHERE symbol = :symbol AND status = 'rewarded' "
                            "AND emitted_at >= :since AND emitted_at < :until"
                        ),
                        {
                            "symbol": args.symbol,
                            "since": w.test_start,
                            "until": w.test_end,
                        },
                    )
                    .mappings()
                    .all()
                )
                net_returns = [
                    float(r["outcome_return_realized"]) - cost_model.total()
                    for r in realized_rows
                    if r["outcome_return_realized"] is not None
                ]
            sharpe = _sharpe(net_returns) if len(net_returns) > 1 else None
            edge_class: str | None = None
            if cost_model.total_bps:
                gross = sum(
                    e.edge.realized_gross * e.n for e in evals if e.edge is not None
                ) / (sum(e.n for e in evals if e.edge is not None) or 1)
                net = gross - cost_model.total()
                edge_class = classify_edge(gross, net, sharpe=sharpe)

            rows.append(
                WfRow(
                    index=w.index,
                    symbol=args.symbol,
                    regime=regime_label,
                    train_start=w.train_start,
                    train_end=w.train_end,
                    test_start=w.test_start,
                    test_end=w.test_end,
                    n_train=sum(s.n for s in train_scores),
                    horizons=evals,
                    accepted=len(accepted),
                    rejected=rejected,
                    cost_bps=cost_model.total_bps,
                    sharpe=sharpe,
                    edge_class=edge_class,
                )
            )
            total_acc = sum(h.model.model_accuracy * h.n for h in evals) / (
                sum(h.n for h in evals) or 1
            )
            total_reward = sum(h.model.model_reward * h.n for h in evals) / (
                sum(h.n for h in evals) or 1
            )
            edge_bits = f" | edge {edge_class}" if edge_class else ""
            print(
                f"    test {regime_label or 'ALL'}: modelo acc "
                f"{total_acc:.1%} reward {total_reward:+.4f} "
                f"n={sum(h.n for h in evals)}{edge_bits}"
            )
            print()

    print(
        render_wf_report(
            rows,
            symbol=args.symbol,
            interval_label=args.interval,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
