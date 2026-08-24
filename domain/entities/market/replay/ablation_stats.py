"""Ablation stats and aggregation (FASE 9.3)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .portfolio import portfolio_net_returns, sharpe_of, max_drawdown

__all__ = [
    "AblationWindow",
    "ablation_window_stats",
    "AblationStats",
    "aggregate_ablation",
]


@dataclass(frozen=True, slots=True)
class AblationWindow:
    """Una ventana × una ablación (nivel TEST real de FASE 9.1/9.2)."""

    symbol: str
    index: int
    regime: str | None
    ablation: str
    cost_bps: int
    n: int
    accuracy: float
    gross_edge: float
    realized_gross: float
    realized_net: float
    sharpe: float
    max_drawdown: float


def ablation_window_stats(
    *,
    symbol: str,
    index: int,
    regime: str | None,
    ablation: str,
    cost_bps: int,
    weights: Mapping[str, float],
    expected: Mapping[str, float],
    accuracy: Mapping[str, float],
    per_timestamp: Sequence[tuple[float, Mapping[str, tuple[bool, float]]]],
) -> AblationWindow:
    """Estadísticas de una ventana × ablación sobre outcomes reales.

    ``gross_edge`` usa el retorno que el modelo DECLARA (señal);
    ``realized_*`` usa el PnL direccional (acierta → +|move|, falla →
    −|move|) que es lo que el mercado PAGÓ. Sharpe y drawdown sobre la
    serie neta por timestamp.
    """
    cost = cost_bps / 10000.0
    net_returns = portfolio_net_returns(weights, per_timestamp, cost)
    realized_gross = (
        sum(net_returns) / len(net_returns) + cost if net_returns else 0.0
    )
    active = {e: w for e, w in weights.items() if w > 0.0}
    gross_edge = sum(w * expected.get(e, 0.0) for e, w in active.items())
    weighted_acc = sum(w * accuracy.get(e, 0.0) for e, w in active.items())
    return AblationWindow(
        symbol=symbol,
        index=index,
        regime=regime,
        ablation=ablation,
        cost_bps=cost_bps,
        n=len(net_returns),
        accuracy=weighted_acc,
        gross_edge=gross_edge,
        realized_gross=realized_gross,
        realized_net=realized_gross - cost,
        sharpe=sharpe_of(net_returns),
        max_drawdown=max_drawdown(net_returns),
    )


@dataclass(frozen=True, slots=True)
class AblationStats:
    """Agregado de una ablación (símbolo, o símbolo × régimen)."""

    ablation: str
    n: int
    accuracy: float
    gross_edge: float
    net_edge: float
    sharpe: float
    max_drawdown: float


def aggregate_ablation(
    windows: Sequence[AblationWindow],
    pooled_returns: Sequence[float],
    window_nets: Sequence[float] | None = None,
) -> AblationStats:
    """Agrega ventanas: edges/accuracy ponderados por n; sharpe de la serie
    por timestamp (pooled en orden cronológico); maxDD del acumulado de
    medias netas POR VENTANA (comparable entre símbolos, no distorsionado
    por el número de trades)."""
    n = sum(w.n for w in windows)
    gross = sum(w.gross_edge * w.n for w in windows) / n if n else 0.0
    net = sum(w.realized_net * w.n for w in windows) / n if n else 0.0
    acc = sum(w.accuracy * w.n for w in windows) / n if n else 0.0
    if window_nets is None:
        window_nets = [w.realized_net for w in windows]
    return AblationStats(
        ablation=windows[0].ablation if windows else "",
        n=n,
        accuracy=acc,
        gross_edge=gross,
        net_edge=net,
        sharpe=sharpe_of(pooled_returns),
        max_drawdown=max_drawdown(window_nets),
    )