#!/usr/bin/env python
"""ZENIN MARKET — OBSERVABILITY & EVALUATION DASHBOARD (FASE 7.5).

Abre ZENIN y responde, con datos reales del store (zenin_market):

- Overview: predicciones / evaluadas / en espera / invalidadas.
- HORIZON: acierto y reward por horizonte (¿cuál horizonte funciona?).
- CONFIDENCE: la curva de calibración — si ZENIN dice P=0.9, ¿históricamente
  acertó ~90% o ~55%? Cada bucket se clasifica OK / INSUFFICIENT /
  ⚠ CALIBRATION FAILURE (no se concluye con muestras diminutas).
- REGIME: rendimiento por régimen (lo que el pipeline registre).
- Curva de calibración ASCII (declarado vs realizado vs diagonal).
- ECE (Expected Calibration Error) y Brier score globales.

Solo lectura: el dashboard jamás modifica el store (append-only).
Los umbrales ``--min-n`` y ``--tolerance`` son el germen de los guardrails
de FASE 8: no aprender con < N observaciones, no aprender por un único
resultado.

Uso:
    python scripts/zenin_dashboard.py
    python scripts/zenin_dashboard.py --symbol NVDA --days 7
    python scripts/zenin_dashboard.py --min-n 10 --tolerance 0.15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_ST_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV, override=True)

from iot_machine_learning.domain.entities.market.replay.calibration import (  # noqa: E402
    BucketStatus,
    CalibrationReport,
    CalibrationThresholds,
    bucket_calibration,
    calibration_chart,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market import (  # noqa: E402
    MarketPredictionRepository,
    ZeninMarketDbConnection,
)

_BOX_WIDTH = 60

_HORIZON_LABELS: dict[int, str] = {
    60: "1m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "1h",
    14400: "4h",
    86400: "1d",
}


def human_horizon(seconds: int) -> str:
    """Etiqueta humana para el horizonte (60 -> '1m', 3600 -> '1h')."""
    return _HORIZON_LABELS.get(seconds, f"{seconds}s")


def _row(line: str) -> str:
    """Ajusta una línea al ancho del box (con bordes)."""
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


def _status_mark(status: BucketStatus) -> str:
    if status is BucketStatus.FAIL:
        return "⚠ CALIBRATION FAILURE"
    if status is BucketStatus.INSUFFICIENT:
        return "⚠ insufficient"
    return "ok"


def render_dashboard(
    stats: dict,
    history: dict,
    report: CalibrationReport,
    *,
    symbol: str = "-",
) -> str:
    """Renderiza el tablero ASCII completo (función pura, testeable).

    Args:
        stats: salida de ``MarketPredictionRepository.overall_stats``.
        history: salida de ``performance_history``.
        report: reporte de calibración (``bucket_calibration``).
    """
    lines: list[str] = []
    lines.append(_top())
    lines.append(_row("ZENIN MARKET"))
    lines.append(_row(f"symbol: {symbol}   source: zenin_market (append-only)"))
    lines.append(_rule())

    predictions = stats["predictions"] or 0
    evaluated = stats["evaluated"] or 0
    pending = stats["pending"] or 0
    invalidated = stats["invalidated"] or 0
    hits = stats["hits"] or 0
    direction_rate = f"{hits / evaluated:.1%}" if evaluated else "-"

    lines.append(_kv("Predictions", f"{predictions}"))
    lines.append(_kv("Evaluated", f"{evaluated}"))
    lines.append(_kv("Pending", f"{pending}"))
    lines.append(_kv("Invalidated", f"{invalidated}"))
    lines.append(_kv("Direction", direction_rate))
    lines.append(_rule())

    lines.append(_header("HORIZON"))
    by_horizon = sorted(
        history["by_horizon"], key=lambda r: r["key"] or 0
    )
    if not by_horizon:
        lines.append(_row("  (sin datos)"))
    for r in by_horizon:
        n = r["evaluated"] or 0
        rate = f"{r['hits'] / n:.1%}" if n else "-"
        reward = f"+{r['reward']:.4f}" if (r["reward"] or 0) >= 0 else f"{r['reward']:.4f}"
        label = human_horizon(int(r["key"] or 0))
        lines.append(_row(f"  {label:<4} {rate:<8} n={n:<4} reward {reward}"))
    lines.append(_rule())

    lines.append(_header("CONFIDENCE (calibración)"))
    for b in report.buckets:
        if b.n == 0:
            lines.append(_row(f"  {b.label:<5} 0/0    --     sin evaluar"))
            continue
        lines.append(
            _row(
                f"  {b.label:<5} {b.hits}/{b.n}  "
                f"{b.hit_rate:.1%}  {_status_mark(b.status)}"
            )
        )
    if not report.buckets:
        lines.append(_row("  (sin datos evaluados)"))
    lines.append(_rule())

    lines.append(_header("REGIME"))
    by_regime = sorted(history["by_regime"], key=lambda r: str(r["key"] or ""))
    if not by_regime:
        lines.append(_row("  (sin datos)"))
    for r in by_regime:
        n = r["evaluated"] or 0
        rate = f"{r['hits'] / n:.1%}" if n else "-"
        lines.append(_row(f"  {str(r['key'] or '-'):<10} {rate:<8} n={n}"))
    lines.append(_bottom())

    ece = report.ece
    brier = stats["brier"]
    lines.append("")
    lines.append("EVALUATION")
    lines.append("----------")
    lines.append(f"  ECE:  {ece:.4f}  (0.0 = calibración perfecta)")
    lines.append(
        f"  Brier: {brier:.4f}" if brier is not None else "  Brier: -"
    )
    lines.append(f"  Reward total: {stats['reward'] or 0.0:+.4f}")
    if report.failing_buckets:
        lines.append(
            "  ⚠ Calibración rota en: "
            + ", ".join(b.label for b in report.failing_buckets)
        )
    for b in report.insufficient_buckets:
        if b.n == 0:
            continue
        lines.append(
            f"  ⚠ {b.label}: n={b.n} < min_n={report.thresholds.min_n} — "
            "no concluye (guardrail FASE 8: no aprender con pocas observaciones)"
        )
    lines.append("")
    lines.append("CALIBRATION CURVE (P(up) declarado vs acierto real)")
    lines.append(calibration_chart(report))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--days", type=int, default=None,
                        help="ventana en días (default: todo el histórico)")
    parser.add_argument("--min-n", type=int, default=5,
                        help="mínimo de observaciones para concluir (guardrail)")
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="|declarado - realizado| máximo tolerado")
    args = parser.parse_args()

    if not ZeninMarketDbConnection.health_check():
        print("MySQL zenin_market no disponible: revisa .env (MYSQL_*) y el "
              "contenedor (docker-compose.yml)")
        return 1

    thresholds = CalibrationThresholds(min_n=args.min_n, tolerance=args.tolerance)
    since = time.time() - args.days * 86400 if args.days else None

    with ZeninMarketDbConnection.get_connection() as conn:
        repo = MarketPredictionRepository(conn)
        stats = repo.overall_stats(symbol=args.symbol, since=since)
        history = repo.performance_history(symbol=args.symbol, since=since)
        samples = [
            (f"{r['bucket']:.1f}", r["avg_probability"], r["hits"] or 0, r["evaluated"] or 0)
            for r in history["by_confidence"]
        ]
        report = bucket_calibration(samples, thresholds)

    print(render_dashboard(stats, history, report, symbol=args.symbol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
