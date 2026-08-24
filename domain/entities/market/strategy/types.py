"""FASE 10.3 — Strategy Types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyCard:
    """Ficha técnica completa de una estrategia."""
    
    strategy: str
    context: str  # "strategy·horizon·regime" o solo "strategy"
    
    # Sample info
    samples: int
    history_days: float
    
    # Direction metrics
    accuracy: float
    wilson_lb: float
    wilson_ub: float  # Upper bound Wilson
    accuracy_ci: tuple[float, float]  # Confidence interval normal
    
    # Economic metrics
    gross_edge: float  # Return promedio antes de costos
    costs: float  # Costos promedio
    net_edge: float  # Return neto
    total_pnl: float  # PnL total acumulado
    
    # Risk metrics
    sharpe_ratio: float
    max_drawdown: float
    volatility: float  # Desviación estándar de returns
    
    # Calibration metrics
    brier_score: float
    ece: float
    
    # Evidence status
    evidence_status: str
    evidence_reason: str
    
    # Statistical significance
    is_significant: bool  # Si es estadísticamente diferente del azar
    p_value: float | None  # p-value de prueba vs azar
    
    @property
    def summary(self) -> str:
        """Resumen de una línea."""
        return f"{self.strategy:<15} n={self.samples:<6} acc={self.accuracy:.2%} net={self.net_edge:+.4f} Sharpe={self.sharpe_ratio:.2f}"
    
    @property
    def is_profitable(self) -> bool:
        """True si net edge > 0."""
        return self.net_edge > 0