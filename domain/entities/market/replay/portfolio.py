"""Portfolio metrics: net returns, Sharpe, max drawdown."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ["portfolio_net_returns", "sharpe_of", "max_drawdown"]


def portfolio_net_returns(
    weights: Mapping[str, float],
    per_timestamp: Sequence[tuple[float, Mapping[str, tuple[bool, float]]]],
    cost: float,
) -> list[float]:
    """Retornos netos del portafolio por timestamp.

    El outcome del mercado (movimiento firmado del horizonte) es
    IDÉNTICO para todos los expertos en el mismo timestamp; la
    estrategia solo aporta ``direction_correct``. El PnL de la apuesta
    direccional del experto e en t es: acierta → +|move|; falla → −|move|.
    El portafolio pesa esos PnL por los pesos de la ablación y resta el
    costo. Solo cuentan timestamps donde TODOS los expertos con peso > 0
    tienen outcome (portafolio totalmente invertido; sin sesgo de media
    inversión).
    """
    active = {e: w for e, w in weights.items() if w > 0.0}
    returns: list[float] = []
    for _, per_expert in per_timestamp:
        if not all(e in per_expert for e in active):
            continue
        pnl = 0.0
        for expert, weight in active.items():
            correct, move = per_expert[expert]
            pnl += weight * (abs(move) if correct else -abs(move))
        returns.append(pnl - cost)
    return returns


def sharpe_of(returns: Sequence[float]) -> float:
    """Media / desviación muestral (ddof=1); 0.0 sin varianza o n < 2."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((v - mean) ** 2 for v in returns) / (len(returns) - 1)
    if var <= 1e-12:
        return 0.0
    std = var**0.5
    return float(mean / std)


def max_drawdown(returns: Sequence[float]) -> float:
    """Peor caída pico→valle de la serie acumulada (negativa; 0 si nunca)."""
    cumulative = 0.0
    peak = 0.0
    worst = 0.0
    for r in returns:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        trough = cumulative - peak
        if trough < worst:
            worst = trough
    return worst