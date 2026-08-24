#!/usr/bin/env python
"""ZENIN MARKET — PREDICTION OBSERVATORY (FASE 10).

La memoria observable de ZENIN:

    Prediction → Outcome → Evaluation → Reward → Dashboard | History

ZENIN aprende → observa → se equivoca → registra → demuestra. Este
script NO toca el cerebro ni escribe nada: lee el store y muestra
dónde el modelo está mintiendo (calibración), cuánto lleva aprendido
(learning curve por contexto) y cuánto le falta para poder confiar en
sí mismo (requerimiento de evidencia: Wilson 95% de la accuracy
acumulada ≥ umbral). Es la primera etapa hacia el derecho a tocar
plata: primero demuestra con memoria.

Uso:
    python scripts/zenin_observatory.py
    python scripts/zenin_observatory.py --symbols NVDA
    python scripts/zenin_observatory.py --expert momentum --horizon 900
    python scripts/zenin_observatory.py --evidence-min-accuracy 0.55
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

from iot_machine_learning.domain.entities.market.observatory import (  # noqa: E402
    ContextLearning,
    ObservationRow,
    calibration_curve,
    dimension_stats,
    evidence_requirement,
    is_degraded,
    learning_curve,
    observatory_summary,
    recency_bands,
    render_observatory,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market import (  # noqa: E402
    ZeninMarketDbConnection,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.market_prediction_repository import (  # noqa: E402
    MarketPredictionRepository,
)

DEFAULT_TARGETS = (20, 100, 500, 1000, 5000, 10000)


def _to_rows(records: tuple[dict, ...]) -> list[ObservationRow]:
    rows: list[ObservationRow] = []
    for r in records:
        rows.append(
            ObservationRow(
                prediction_id=str(r["prediction_id"]),
                emitted_at=float(r["emitted_at"]),
                strategy=str(r.get("strategy") or "?"),
                horizon_seconds=int(r["horizon_seconds"]),
                regime=r.get("regime"),
                probability_up=float(r["probability_up"]),
                direction_correct=(
                    bool(r["direction_correct"])
                    if r.get("direction_correct") is not None
                    else None
                ),
                outcome_return_realized=(
                    float(r["outcome_return_realized"])
                    if r.get("outcome_return_realized") is not None
                    else None
                ),
                reward_total=(
                    float(r["reward_total"])
                    if r.get("reward_total") is not None
                    else None
                ),
                calibration_error=(
                    float(r["calibration_error"])
                    if r.get("calibration_error") is not None
                    else None
                ),
                status=str(r["status"]),
                data_status=r.get("data_status"),
            )
        )
    return rows


def _contexts(
    rows: list[ObservationRow],
    *,
    targets: tuple[int, ...],
    min_accuracy: float,
    step: int,
    min_n: int,
) -> list[ContextLearning]:
    groups: dict[tuple[str, int, str], list[ObservationRow]] = {}
    for row in rows:
        regime = row.regime or "ALL"
        groups.setdefault((row.strategy, row.horizon_seconds, regime), []).append(row)
    contexts: list[ContextLearning] = []
    for (strategy, horizon, regime), group in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        if len(group) < min_n:
            continue
        contexts.append(
            ContextLearning(
                label=f"{strategy} · {horizon}s · {regime}",
                points=learning_curve(group, targets=targets),
                requirement=evidence_requirement(
                    group,
                    min_accuracy=min_accuracy,
                    step=step,
                ),
            )
        )
    return contexts


def _render_symbol(
    symbol: str,
    rows: list[ObservationRow],
    *,
    targets: tuple[int, ...],
    evidence_min_accuracy: float,
    evidence_step: int,
    min_context_n: int,
    bands: int,
) -> str:
    summary = observatory_summary(rows)
    calibration = calibration_curve(rows)
    contexts = _contexts(
        rows,
        targets=targets,
        min_accuracy=evidence_min_accuracy,
        step=evidence_step,
        min_n=min_context_n,
    )
    band_stats = recency_bands(rows, bands=bands)
    return render_observatory(
        symbol=symbol,
        summary=summary,
        by_horizon=dimension_stats(rows, key=lambda r: f"{r.horizon_seconds}s"),
        by_strategy=dimension_stats(rows, key=lambda r: r.strategy),
        by_regime=dimension_stats(rows, key=lambda r: r.regime or "ALL"),
        calibration=calibration,
        contexts=contexts,
        bands=band_stats,
        degraded=is_degraded(band_stats),
        evidence_min_accuracy=evidence_min_accuracy,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="NVDA,AMD,AAPL,BTC-USD")
    parser.add_argument("--since", type=float, default=None)
    parser.add_argument("--until", type=float, default=None)
    parser.add_argument("--expert", default=None, help="filtrar por estrategia")
    parser.add_argument("--horizon", type=int, default=None,
                        help="filtrar por horizonte en segundos")
    parser.add_argument("--regime", default=None,
                        help="filtrar por régimen (TRENDING/RANGE/HIGH_VOL/CRASH)")
    parser.add_argument("--calibration-min-n", type=int, default=20)
    parser.add_argument("--calibration-tolerance", type=float, default=0.10)
    parser.add_argument("--evidence-min-accuracy", type=float, default=0.52,
                        help="umbral de Wilson 95% para considerar un contexto")
    parser.add_argument("--evidence-step", type=int, default=20)
    parser.add_argument("--learning-targets", default=",".join(map(str, DEFAULT_TARGETS)),
                        help="objetivos de n para la learning curve")
    parser.add_argument("--min-context-n", type=int, default=20,
                        help="n mínimo para mostrar un contexto en la learning curve")
    parser.add_argument("--bands", type=int, default=4)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("sin símbolos: --symbols NVDA,AMD,AAPL,BTC-USD")
        return 1

    targets = tuple(
        int(t) for t in args.learning_targets.split(",") if t.strip()
    )
    if not targets:
        print("--learning-targets vacío")
        return 1

    if not ZeninMarketDbConnection.health_check():
        print("MySQL zenin_market no disponible: revisa .env (MYSQL_*)")
        return 1

    filters = {
        "since": args.since,
        "until": args.until,
        "status": None,
    }
    out: list[str] = []
    with ZeninMarketDbConnection.get_connection() as conn:
        repo = MarketPredictionRepository(conn)
        for symbol in symbols:
            records = repo.prediction_records(symbol=symbol, **filters)
            rows = _to_rows(records)
            if args.expert:
                rows = [r for r in rows if r.strategy == args.expert]
            if args.horizon:
                rows = [r for r in rows if r.horizon_seconds == args.horizon]
            if args.regime:
                rows = [r for r in rows if (r.regime or "ALL") == args.regime]
            out.append(
                _render_symbol(
                    symbol,
                    rows,
                    targets=targets,
                    evidence_min_accuracy=args.evidence_min_accuracy,
                    evidence_step=args.evidence_step,
                    min_context_n=args.min_context_n,
                    bands=args.bands,
                )
            )
            out.append("")
            out.append("")

    print("\n".join(out).rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
