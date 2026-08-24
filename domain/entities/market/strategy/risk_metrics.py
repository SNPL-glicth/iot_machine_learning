"""FASE 10.3 — Risk Metrics (Sharpe, Max DD, Confidence Intervals)."""

from __future__ import annotations

import math


def compute_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    """Calcula Sharpe ratio."""
    if not returns or len(returns) < 2:
        return 0.0
    
    excess_returns = [r - risk_free_rate for r in returns]
    mean_excess = sum(excess_returns) / len(excess_returns)
    
    if len(excess_returns) < 2:
        return 0.0
    
    variance = sum((r - mean_excess) ** 2 for r in excess_returns) / (len(excess_returns) - 1)
    std = math.sqrt(variance)
    
    if std == 0:
        return 0.0
    
    return mean_excess / std


def compute_max_drawdown(returns: list[float]) -> float:
    """Calcula Maximum Drawdown."""
    if not returns:
        return 0.0
    
    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0
    
    for r in returns:
        cumulative *= (1 + r)
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / peak
        if dd > max_dd:
            max_dd = dd
    
    return max_dd


def compute_confidence_interval(
    values: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calcula intervalo de confianza usando t-distribution."""
    if len(values) < 2:
        return (values[0] if values else 0.0, values[0] if values else 0.0)
    
    n = len(values)
    mean = sum(values) / n
    
    # Desviación estándar muestral
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    
    # t-score para 95% confidence (aproximación normal para n > 30)
    if n > 30:
        z = 1.96  # 95% confidence normal
    else:
        # t-distribution (simplificado)
        z = 2.0  # Approximation
    
    margin = z * std / math.sqrt(n)
    return (mean - margin, mean + margin)