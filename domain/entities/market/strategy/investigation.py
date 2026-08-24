"""FASE 10.3 — Strategy Investigation: fichas técnicas con estadísticas robustas.

Objetivo:
- Crear fichas técnicas por estrategia con métricas estadísticas
- No "vi 52%, entonces funciona" sino análisis con intervalos de confianza
- Incluir dimensiones económicas (gross edge, costs, net edge, Sharpe, max DD)
- Conectar con calibración y evidencia del Evidence Engine
- Comparación estadística entre estrategias

Ficha técnica incluye:
- Samples: n total por estrategia
- Accuracy: con intervalo de confianza (Wilson 95%)
- Economic metrics: gross edge, costs, net edge
- Risk metrics: Sharpe ratio, Maximum Drawdown
- Calibration: ECE, Brier score
- Evidence: estado del Evidence Engine
- Statistical significance: pruebas de hipótesis
"""

from __future__ import annotations

from .risk_metrics import (
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_confidence_interval,
)
from .types import StrategyCard
from .investigator import StrategyInvestigator
from .render import render_strategy_card, render_strategy_comparison

__all__ = [
    "StrategyCard",
    "StrategyInvestigator",
    "compute_sharpe_ratio",
    "compute_max_drawdown",
    "compute_confidence_interval",
    "render_strategy_card",
    "render_strategy_comparison",
]