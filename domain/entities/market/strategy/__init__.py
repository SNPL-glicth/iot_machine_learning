"""FASE 10.3 — Strategy Investigation: fichas técnicas con estadísticas robustas.

Este módulo implementa la investigación de estrategias con análisis
estadístico completo, no solo "vi 52%, entonces funciona":

- StrategyCard: ficha técnica completa con métricas multidimensionales
- StrategyInvestigator: investigador que produce fichas desde datos
- Métricas estadísticas: intervalos de confianza, pruebas de significancia
- Métricas económicas: gross edge, costs, net edge, Sharpe, Max DD
- Conexión con calibración y Evidence Engine

Enfoque: cada estrategia debe demostrar rentabilidad estadísticamente
significativa para sobrevivir, no solo accuracy por encima del azar.
"""

from .investigation import (
    StrategyCard,
    StrategyInvestigator,
    compute_confidence_interval,
    compute_max_drawdown,
    compute_sharpe_ratio,
    render_strategy_card,
    render_strategy_comparison,
)

__all__ = [
    "StrategyCard",
    "StrategyInvestigator",
    "compute_confidence_interval",
    "compute_max_drawdown",
    "compute_sharpe_ratio",
    "render_strategy_card",
    "render_strategy_comparison",
]