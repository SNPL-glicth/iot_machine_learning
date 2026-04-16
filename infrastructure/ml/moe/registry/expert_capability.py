"""ExpertCapability — declaración de capacidades de un experto.

Dataclass inmutable que describe qué puede hacer un experto.
Usado por ExpertRegistry para matching de contexto.

SRP: Solo describe capacidades, no implementa lógica de matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class ExpertCapability:
    """Capacidades declaradas de un experto MoE.
    
    Define los contextos en los que el experto puede operar efectivamente.
    Usado por ExpertRegistry.find_by_regime() y get_candidates().
    
    Attributes:
        regimes: Regímenes soportados (stable, trending, volatile, noisy).
        domains: Dominios de aplicación (iot, finance, healthcare).
        min_points: Mínimo de puntos requeridos para operar.
        max_points: Máximo recomendado (0 = sin límite).
        specialties: Especialidades específicas (ej: seasonality, anomalies).
        computational_cost: Costo relativo (1.0 = baseline).
        
    Example:
        >>> caps = ExpertCapability(
        ...     regimes=("volatile", "trending"),
        ...     domains=("iot",),
        ...     min_points=5,
        ...     computational_cost=2.0
        ... )
    """
    
    regimes: Tuple[str, ...] = ("stable",)
    domains: Tuple[str, ...] = ("iot",)
    min_points: int = 3
    max_points: int = 0  # 0 = ilimitado
    specialties: Tuple[str, ...] = ()
    computational_cost: float = 1.0
    
    def __post_init__(self):
        """Validar invariants."""
        if self.min_points < 0:
            raise ValueError(f"min_points must be non-negative, got {self.min_points}")
        if self.max_points < 0:
            raise ValueError(f"max_points must be non-negative, got {self.max_points}")
        if self.computational_cost <= 0:
            raise ValueError(f"computational_cost must be positive, got {self.computational_cost}")
    
    def supports_regime(self, regime: str) -> bool:
        """Check if capability supports given regime."""
        return regime in self.regimes
    
    def supports_domain(self, domain: str) -> bool:
        """Check if capability supports given domain."""
        return domain in self.domains
    
    def can_handle_points(self, n_points: int) -> bool:
        """Check if n_points is within acceptable range."""
        if n_points < self.min_points:
            return False
        if self.max_points > 0 and n_points > self.max_points:
            return False
        return True
    
    def matches_context(self, regime: str, domain: str, n_points: int) -> bool:
        """Full context matching."""
        return (
            self.supports_regime(regime) and
            self.supports_domain(domain) and
            self.can_handle_points(n_points)
        )
