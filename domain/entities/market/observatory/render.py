"""ZENIN — Prediction Observatory: render del dashboard (FASE 10).

Convierte los agregados puros de ``observatory.py`` en el dashboard
ASCII que el usuario lee y archiva. No toca el store: es presentación
pura, sin estado.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..replay.calibration import BucketStatus, CalibrationReport
from .observatory import BandStat, ContextLearning, DimensionStat, ObservatorySummary

__all__ = ["render_observatory"]

_BAR_WIDTH = 20


def _bar(accuracy: float, width: int = _BAR_WIDTH) -> str:
    filled = round(max(0.0, min(1.0, accuracy)) * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_count(value: int) -> str:
    return f"{value:,}"


def _render_summary(summary: ObservatorySummary) -> str:
    total = summary.total
    lines = [
        "TOTAL PREDICTIONS",
        "-" * 15,
        _fmt_count(total),
        "",
        "EVALUATED",
        "-" * 9,
        _fmt_count(summary.evaluated),
        "",
        "PENDING",
        "-" * 7,
        _fmt_count(summary.pending),
        "",
    ]
    extras: list[str] = []
    if summary.invalidated:
        extras.append(f"INVALIDATED {_fmt_count(summary.invalidated)}")
    if summary.archived:
        extras.append(f"ARCHIVED {_fmt_count(summary.archived)}")
    if summary.stale:
        extras.append(f"STALE {_fmt_count(summary.stale)}")
    if extras:
        lines.append("·".join(extras))
        lines.append("")
    return "\n".join(lines)


def _render_dimension(title: str, stats: Sequence[DimensionStat]) -> str:
    if not stats:
        return f"{title}\n" + "-" * len(title) + "\n(sin datos)\n"
    lines = [title, "-" * len(title)]
    for s in stats:
        suffix = ""
        if s.n:
            suffix = f"  n={_fmt_count(s.n)}  Wilson≥{_fmt_pct(s.wilson_lb)}"
        lines.append(
            f"{s.label:<10} {_fmt_pct(s.accuracy)}  {_bar(s.accuracy)}"
            f"{suffix}"
        )
    return "\n".join(lines)


def _render_calibration(report: CalibrationReport) -> str:
    lines = [
        "PROBABILITY CALIBRATION",
        "-" * 22,
        f"declarado vs realizado por bucket · ECE={_fmt_pct(report.ece)}"
        f" · FAIL si |δ|>{_fmt_pct(report.thresholds.tolerance)}",
        "",
        "bucket  declarado  realizado   |δ|   estado",
    ]
    for b in report.buckets:
        marker = ""
        if b.status is BucketStatus.FAIL:
            if b.delta > 0:
                marker = "  <== SOBRECONFIANZA (declara más de lo que acierta)"
            else:
                marker = "  <== SUBESTIMACIÓN (declara menos de lo que acierta)"
        elif b.status is BucketStatus.INSUFFICIENT:
            marker = f"  (n < {report.thresholds.min_n}, no concluye)"
        lines.append(
            f"{b.label:<6} {_fmt_pct(b.declared):>8}   "
            f"{_fmt_pct(b.hit_rate):>8}   {_fmt_pct(abs(b.delta)):>6}   "
            f"{b.status.value}{marker}"
        )
    return "\n".join(lines)


def _render_contexts(
    contexts: Sequence[ContextLearning],
    *,
    min_accuracy: float,
) -> str:
    lines = [
        "LEARNING CURVE — accuracy acumulada por contexto",
        "-" * 44,
        f"contexto = experto × horizonte × régimen · "
        f"suficiente = Wilson 95% ≥ {_fmt_pct(min_accuracy)}",
        "",
    ]
    for ctx in contexts:
        lines.append(f"--- {ctx.label}")
        if not ctx.points:
            lines.append("    (sin observaciones evaluadas)")
            continue
        for p in ctx.points:
            enough = (
                ctx.requirement is not None and p.n >= ctx.requirement
            )
            lines.append(
                f"    n={p.n:>6}  {_fmt_pct(p.accuracy):>7}"
                f"  Wilson≥{_fmt_pct(p.wilson_lb):>7}"
                f"  {'suficiente' if enough else 'insuficiente'}"
            )
        if ctx.requirement is not None:
            lines.append(
                f"    evidencia: necesita ≈ {_fmt_count(ctx.requirement)}"
                f" observaciones (Wilson 95% ≥ {_fmt_pct(min_accuracy)})"
            )
        else:
            max_wilson = (
                max(p.wilson_lb for p in ctx.points) if ctx.points else 0.0
            )
            lines.append(
                f"    evidencia: aún no — máx Wilson observado "
                f"{_fmt_pct(max_wilson)} < {_fmt_pct(min_accuracy)}"
            )
        lines.append("")
    return "\n".join(lines)


def _render_bands(bands: Sequence[BandStat], *, degraded: bool) -> str:
    lines = [
        "RECENCIA — accuracy por banda (antigua → reciente)",
        "-" * 46,
    ]
    if not bands:
        lines.append("(sin datos evaluados)")
        return "\n".join(lines)
    for b in bands:
        lines.append(
            f"    banda {b.band + 1}/{len(bands)}  {_fmt_pct(b.accuracy):>7}"
            f"  Wilson≥{_fmt_pct(b.wilson_lb):>7}"
            f"  n={_fmt_count(b.n)}"
        )
    if degraded:
        lines.append("    ESTADO: DEGRADADA — la banda reciente es peor que la antigua")
    else:
        lines.append("    ESTADO: estable")
    return "\n".join(lines)


def render_observatory(
    *,
    symbol: str,
    summary: ObservatorySummary,
    by_horizon: Sequence[DimensionStat],
    by_strategy: Sequence[DimensionStat],
    by_regime: Sequence[DimensionStat],
    calibration: CalibrationReport,
    contexts: Sequence[ContextLearning],
    bands: Sequence[BandStat],
    degraded: bool,
    evidence_min_accuracy: float,
) -> str:
    """Dashboard completo en ASCII (puro, sin estado)."""
    sections = [
        f"ZENIN — PREDICTION OBSERVATORY — {symbol}",
        "=" * len(f"ZENIN — PREDICTION OBSERVATORY — {symbol}"),
        "",
        "ESTADO GLOBAL",
        "-" * 13,
        f"accuracy={_fmt_pct(summary.accuracy)}"
        f" (Wilson≥{_fmt_pct(summary.wilson_lb)})"
        f" · reward medio={summary.mean_reward:+.4f}"
        f" · calibración media={_fmt_pct(abs(summary.mean_calibration_error))}",
        "",
        _render_summary(summary),
        _render_dimension("BY HORIZON", by_horizon),
        "",
        _render_dimension("BY STRATEGY", by_strategy),
        "",
        _render_dimension("BY REGIME", by_regime),
        "",
        _render_calibration(calibration),
        "",
        _render_contexts(contexts, min_accuracy=evidence_min_accuracy),
        _render_bands(bands, degraded=degraded),
    ]
    return "\n".join(sections)
