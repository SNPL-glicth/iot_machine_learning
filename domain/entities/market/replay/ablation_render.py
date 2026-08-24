"""Ablation matrix rendering (FASE 9.3)."""

from __future__ import annotations

from collections.abc import Mapping

from .ablation_constants import ABLATIONS
from .ablation_stats import AblationStats

__all__ = ["render_ablation_matrix"]

_HEADER = f"{'Ablación':<18}{'n':>7}{'acc':>7}{'gross':>9}{'net':>9}{'sharpe':>8}{'maxDD':>9}"


def _stats_cells(stats: AblationStats | None) -> str:
    if stats is None or stats.n == 0:
        return f"{'-':>7}{'-':>7}{'-':>9}{'-':>9}{'-':>8}{'-':>9}"
    return (
        f"{stats.n:>7}"
        f"{stats.accuracy:.1%}"
        f"{stats.gross_edge:+.2%}"
        f"{stats.net_edge:+.2%}"
        f"{stats.sharpe:+.2f}"
        f"{stats.max_drawdown:+.2%}"
    )


def render_ablation_matrix(
    stats_by_symbol: Mapping[str, Mapping[str, AblationStats]],
    by_regime: Mapping[str, Mapping[tuple[str, str], AblationStats]],
    *,
    cost_bps: Mapping[str, int],
    window_counts: Mapping[str, int],
) -> str:
    """Matriz por símbolo (todas las ventanas) + desglose por régimen."""
    out: list[str] = []
    out.append("ABLATION MATRIX — ZENIN MARKET (FASE 9.3)")
    out.append("la pregunta: ¿de dónde salió el edge bruto?")
    out.append("")
    for symbol in stats_by_symbol:
        symbol_stats = stats_by_symbol[symbol]
        out.append(
            f"== {symbol} ({cost_bps.get(symbol, 0)} bps, "
            f"{window_counts.get(symbol, 0)} ventanas) =="
        )
        out.append(_HEADER)
        for ablation in ABLATIONS:
            stats = symbol_stats.get(ablation)
            if stats is None or stats.n == 0:
                continue
            out.append(f"{ablation:<18}{_stats_cells(stats)}")
        regime_map = by_regime.get(symbol, {})
        if regime_map:
            out.append("  por régimen:")
            for (ablation, regime), stats in sorted(
                regime_map.items(), key=lambda item: (item[0][1], ABLATIONS.index(item[0][0]))
            ):
                out.append(
                    f"    {regime:<8}{ablation:<18}{_stats_cells(stats)}"
                )
        out.append("")
    return "\n".join(out)