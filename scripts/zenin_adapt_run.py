#!/usr/bin/env python
"""ZENIN MARKET — ADAPTATION PROPOSAL MODE (FASE 8).

PROPUESTA ≠ CAMBIO. Este script NO toca el modelo hasta que el guardrail
lo autorice, y todo —propuesta, veredicto, razón— queda registrado:

    Historial (solo outcomes reales)
        → PerformanceAnalyzer → ExpertScores
        → WeightProposer → WeightProposal (impresión antes de actuar)
        → AdaptationGuard → ACCEPT / REJECT (10 chequeos auditables)
        → model_versions: v2 = v1 + propuestas aceptadas (append-only)

Regla de piedra: ZENIN nunca aprende de su propia predicción sin haber
observado el outcome externo. El análisis consume EXCLUSIVAMENTE filas
evaluadas (status=rewarded, sin INVALIDATED ni STALE).

Cada experto corre su propio shadow live (mismo fragmento, mismo feed,
misma ventana) y sus predicciones se persisten con strategy=experto;
después el store decide con números reales quién merece más peso.

Uso:
    python scripts/zenin_adapt_run.py
    python scripts/zenin_adapt_run.py --symbol NVDA --start 09:30 --end 10:30
    python scripts/zenin_adapt_run.py --experts naive,momentum --min-n 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

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
from iot_machine_learning.domain.entities.market.replay import (  # noqa: E402
    LiveClock,
    ReplayEngineConfig,
)
from iot_machine_learning.domain.entities.market.replay.baselines import (  # noqa: E402
    BASELINES,
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
    AdaptationRepository,
    MarketPredictionRepository,
    ZeninMarketDbConnection,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.migrations import (  # noqa: E402, F401
    apply_migrations,
)

_BOX_WIDTH = 60


def _row(line: str) -> str:
    return "║ " + line.ljust(_BOX_WIDTH - 4) + " ║"


def _rule(char: str = "═") -> str:
    return "╠" + char * (_BOX_WIDTH - 2) + "╣"


def _top() -> str:
    return "╔" + "═" * (_BOX_WIDTH - 2) + "╗"


def _bottom() -> str:
    return "╚" + "═" * (_BOX_WIDTH - 2) + "╝"


def _header(title: str) -> str:
    return _row(title)


def _kv(label: str, value: str) -> str:
    return _row(f"{label:<12} {value:>{_BOX_WIDTH - 17}}")

BANNER = """
██╗   ██╗███████╗███╗   ██╗██╗███╗   ██╗   ██████╗  ██████╗  ██████╗  ██████╗
██║   ██║██╔════╝████╗  ██║██║████╗  ██║   ╚════██╗██╔═████╗██╔═████╗╚════██╗
██║   ██║█████╗  ██╔██╗ ██║██║██╔██╗ ██║    █████╔╝██║██╔██║██║██╔██║ █████╔╝
╚██╗ ██╔╝██╔══╝  ██║╚██╗██║██║██║╚██╗██║    ╚═══██╗████╔╝██║████╔╝██║ ╚═══██╗
 ╚████╔╝ ███████╗██║ ╚████║██║██║ ╚████║   ██████╔╝╚██████╔╝╚██████╔╝╚██████╔╝
  ╚═══╝  ╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝   ╚═════╝  ╚═════╝  ╚═════╝  ╚═════╝
"""


def _card(proposal) -> str:
    """Tarjeta de propuesta: se IMPRIME y se GUARDA antes de actuar."""
    lines = [
        _top(),
        _header(f"PROPOSAL — {proposal.context_label}"),
        _rule(),
        _kv("Expert", proposal.expert),
        _kv("Regime", proposal.regime or "-"),
        _kv("Horizon", f"{proposal.horizon_seconds}s"),
        _kv("Current weight", f"{proposal.current_weight:.3f}"),
        _kv("Observed reward", f"{proposal.observed_reward:+.4f}"),
        _kv("Calibration", f"{proposal.calibration:.3f}"),
        _kv("Sample size", f"{proposal.sample_size}"),
        _kv("Proposed weight", f"{proposal.proposed_weight:.3f}"),
        _kv("Delta", f"{proposal.weight_delta:+.4f}"),
        _rule(),
        _row(f'Reason: "{proposal.reason}"'),
        _bottom(),
    ]
    return "\n".join(lines)


def _weights_snapshot(rows: tuple[dict, ...]) -> dict[str, dict[str, float]]:
    """Contextos con muestra suficiente -> pesos uniformes iniciales."""
    seen: dict[tuple[str | None, int], list[str]] = {}
    for row in rows:
        context = (row["regime"], row["horizon_seconds"])
        seen.setdefault(context, []).append(str(row["strategy"]))
    return {
        f"*|{regime or '-'}|{horizon}s": default_weights(expert_names)
        for (regime, horizon), expert_names in seen.items()
    }


def _version_reason(accepted: list) -> str:
    if not accepted:
        return "no hay cambios: ninguna propuesta superó los guardrails"
    lines = [f"Adaptación aceptada para {len(accepted)} propuesta(s):"]
    for p in sorted(accepted, key=lambda p: p.context_label):
        lines.append(
            f"  - {p.expert}@{p.regime or '-'}/{p.horizon_seconds}s: "
            f"{p.current_weight:.3f} -> {p.proposed_weight:.3f} ({p.reason})"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS), default="1m")
    parser.add_argument("--start", default="09:30", help="inicio fragmento (UTC)")
    parser.add_argument("--end", default="10:30", help="fin fragmento (UTC)")
    parser.add_argument(
        "--drop", action="append", default=[], help="ventana de caída HH:MM:SS-HH:MM:SS (repetible)"
    )
    parser.add_argument(
        "--experts",
        default="naive,momentum,ema-crossover,mean-reversion",
        help="expertos a correr (separados por coma)",
    )
    parser.add_argument("--min-n", type=int, default=10, help="muestra mínima por contexto")
    parser.add_argument(
        "--max-change", type=float, default=0.10, help="|Δ peso| máximo por experto y versión"
    )
    parser.add_argument(
        "--min-history-days", type=int, default=2, help="días distintos observados por contexto"
    )
    parser.add_argument("--calibration-penalty", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--parent", type=int, default=None, help="version_id padre (default: la activa)"
    )
    args = parser.parse_args()

    experts = {p.name: p for p in BASELINES}
    requested = [name.strip() for name in args.experts.split(",") if name.strip()]
    unknown = [name for name in requested if name not in experts]
    if unknown:
        print(f"experto(s) desconocido(s): {unknown}")
        print(f"disponibles: {sorted(experts)}")
        return 1
    if "random" in requested:
        print("random no es un experto aprendible: se excluye del análisis")
        requested.remove("random")

    print(BANNER)
    print()
    print("FASE 8 — ADAPTATION PROPOSAL MODE")
    print("PROPUESTA ≠ CAMBIO: este script NO modifica el modelo sin que el")
    print("guardrail lo autorice; y todo —propuesta, veredicto, razón— queda")
    print("registrado en zenin_market (append-only).")
    print()

    interval, horizons = RESOLUTIONS[args.resolution]
    path = (
        Path(__file__).resolve().parent.parent
        / "data" / "market" / f"{args.symbol}_{args.resolution}.csv"
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
    print()

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
    feed = DropWindowsFeed(fragment, drops) if drops else fragment

    print(
        f"Fragmento: {fmt_ts(start_ts)} -> {fmt_ts(end_ts)} (UTC) | "
        f"instrumento: {args.symbol} | resolución: {interval}s | "
        f"horizontes: {horizons}"
    )
    print(f"Expertos: {requested}")
    print()

    # 1) Cada experto corre su propio shadow live sobre el MISMO fragmento.
    with ZeninMarketDbConnection.get_connection() as conn:
        repo = MarketPredictionRepository(conn)
        for name in requested:
            live_feed = LiveFeed(
                symbol=args.symbol,
                historical_feed=feed,
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
                    predictor=experts[name],
                ),
            ).run()
            predictions = list(shadow.all_predictions)
            written = repo.save_batch(predictions)
            print(
                f"  {name:<14} predicciones emitidas: {len(predictions):<4} "
                f"persistidas: {written} (strategy={name})"
            )
        print()

        # 2) Performance Analyzer sobre SOLO outcomes reales.
        rows = repo.expert_performance(symbol=args.symbol)
        analyzer = PerformanceAnalyzer(calibration_penalty=args.calibration_penalty)
        scores = analyzer.analyze(rows)
        print("PERFORMANCE ANALYZER (solo filas evaluadas: rewardeds, sin INVALIDATED/STALE)")
        print("--------------------------------------------------------------")
        for s in sorted(scores, key=lambda s: (s.regime or "", s.horizon_seconds, s.expert)):
            print(
                f"  {s.expert:<14} {s.regime or 'ALL':<10} "
                f"{s.horizon_seconds:>5}s  n={s.n:<4} acc={s.accuracy:>6.1%} "
                f"reward={s.mean_reward:+.4f} cal={s.calibration_error:.2f} "
                f"adj={s.reward_adjusted:+.4f}  días={s.history_days}"
            )
        if not scores:
            print("  (sin historial evaluado: no hay nada que proponer)")
        print()

        # 3) Pesos actuales: la versión activa del modelo (o defaults).
        adaptation = AdaptationRepository(conn)
        latest = adaptation.latest_version()
        if latest is not None:
            current_weights: dict[str, dict[str, float]] = {
                str(ctx): dict(weights) for ctx, weights in json.loads(latest["weights"]).items()
            }
        else:
            current_weights = _weights_snapshot(rows)
            # Bootstrap: la primera corrida materializa v1 (pesos uniformes)
            # para que la cadena parent_version esté completa desde el inicio.
            version_id = adaptation.create_version(
                weights=current_weights,
                calibration={},
                reason="v1 inicial: pesos uniformes por contexto (bootstrap)",
                created_at=time.time(),
            )
            latest = adaptation.version(version_id)
        parent_label = f"v{latest['version_id']}"
        if not current_weights:
            print("Modelo actual: (sin contextos con datos)")
        else:
            print(f"Modelo actual: {parent_label}")
            for label, weights in sorted(current_weights.items()):
                print(
                    f"  {label:<20} "
                    + "  ".join(f"{k}={v:.3f}" for k, v in sorted(weights.items()))
                )
        print()

        # 4) WeightProposer: propuestas por contexto (NO toca el modelo).
        proposer = WeightProposer(
            min_n=args.min_n,
            max_change=args.max_change,
            temperature=args.temperature,
        )
        proposals = proposer.propose(
            scores,
            current_weights,
            parent_version=(str(latest["version_id"]) if latest is not None else None),
        )
        print(f"PROPOSICIONES ({len(proposals)}) — se imprimen y se guardan ANTES de actuar")
        print("--------------------------------------------------------------")
        by_context: dict[tuple[str | None, int], list] = {}
        for s in scores:
            by_context.setdefault((s.regime, s.horizon_seconds), []).append(s)
        vectors: dict[tuple[str | None, int], dict[str, float]] = {}
        for context, group in by_context.items():
            vector, _ = proposer.propose_vector(context[0], context[1], group, current_weights)
            vectors[context] = vector
        print()

        # 5) AdaptationGuard: 10 chequeos por propuesta -> ACCEPT / REJECT.
        guard = AdaptationGuard(
            min_n=args.min_n,
            min_history_days=args.min_history_days,
            max_change=args.max_change,
        )
        accepted: list = []
        verdicts: dict[str, object] = {}
        for proposal in sorted(proposals, key=lambda p: p.context_label):
            score = next(
                s for s in scores if (s.expert, s.regime, s.horizon_seconds)
                == (proposal.expert, proposal.regime, proposal.horizon_seconds)
            )
            result = guard.evaluate(
                proposal,
                history_days=score.history_days,
                data_quality="clean",
                context_weights_after=vectors[(proposal.regime, proposal.horizon_seconds)],
            )
            verdicts[proposal.context_label] = result
            print(_card(proposal))
            print()
            print("GUARDRAIL")
            print("---------")
            print(result.render())
            print()
            proposal_id = adaptation.record_proposal(proposal, result)
            print(
                f"  registrada: adaptation_proposals/{proposal_id} "
                f"({'accepted' if result.passed else 'rejected'})"
            )
            print()
            if result.passed:
                accepted.append(proposal)
        print()

        # 6) Si algo pasa el guardrail: nueva versión (append-only).
        if accepted:
            new_weights = dict(current_weights)
            for proposal in accepted:
                key = f"*|{proposal.regime or '-'}|{proposal.horizon_seconds}s"
                new_weights[key] = dict(vectors[(proposal.regime, proposal.horizon_seconds)])
            calibration_snapshot = {
                s.context_label: {
                    "calibration_error": s.calibration_error,
                    "reward_adjusted": s.reward_adjusted,
                    "n": s.n,
                }
                for s in scores
            }
            version_id = adaptation.create_version(
                weights=new_weights,
                calibration=calibration_snapshot,
                reason=_version_reason(accepted),
                proposal_ids=[p.context_label for p in accepted],
                guard_checks=[
                    {
                        "proposal": p.context_label,
                        "checks": [
                            {
                                "name": c.name,
                                "ok": c.ok,
                                "detail": c.detail,
                            }
                            for c in verdicts[p.context_label].checks
                        ],
                    }
                    for p in accepted
                ],
                parent_version_id=args.parent,
            )
            print(
                f"✓ MODEL v{version_id} creada (parent={parent_label}, "
                f"append-only). La versión anterior queda preservada "
                f"(parent_version_id={latest['version_id']})."
            )
            print()
            print("VERSION DIFF")
            print("------------")
            for key, new in sorted(new_weights.items()):
                old = current_weights.get(key, {})
                changes = [
                    f"{k}: {old.get(k, 0.0):.3f} -> {v:.3f}"
                    for k, v in sorted(new.items())
                    if abs(old.get(k, 0.0) - v) > 1e-9
                ]
                if changes:
                    print(f"  {key:<20} " + ", ".join(changes))
            print()
            print("AUDIT TRAIL")
            print("-----------")
            for v in adaptation.list_versions(limit=5):
                print(
                    f"  v{v['version_id']}  is_active={v['is_active']}  "
                    f"parent={v['parent_version_id']}  propuestas="
                    f"{v['proposal_id'] or '-'}  creada={fmt_ts(v['created_at'])}"
                )
        else:
            print(
                "Sin cambios: ninguna propuesta superó los guardrails. "
                "El modelo se mantiene (v1 intacta)."
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
