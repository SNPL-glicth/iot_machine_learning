"""
Watchdog para detectar ensemble collapse y activar recovery forzado.
Observa estado global de pesos/supresión sin modificar componentes existentes.

Principio: Observer pattern — watchdog observa, no interfiere directamente.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging


class EnsembleHealth(Enum):
    """Estado de salud del ensemble."""
    HEALTHY = "healthy"           # >50% engines activos con peso significativo
    DEGRADED = "degraded"         # 30-50% engines activos — warning
    CRITICAL = "critical"         # <30% engines activos — recovery inminente
    COLLAPSED = "collapsed"       # 0 engines con peso > min_weight — recovery forzado


@dataclass
class WatchdogSnapshot:
    """Estado del ensemble en un momento dado."""
    total_engines: int
    active_engines: int        # peso > min_weight_threshold
    suppressed_engines: int    # supresión > suppression_threshold
    min_weight: float          # peso mínimo actual en ensemble
    max_suppression: float     # supresión máxima actual
    health: EnsembleHealth
    recovery_recommended: bool
    reason: str


class EnsembleWatchdog:
    """
    Observa salud del ensemble y detecta collapse.
    
    NO modifica pesos directamente — solo detecta y reporta.
    La acción de recovery la ejecuta ForcedRecoveryManager.
    
    Parámetros:
    - min_active_ratio: fracción mínima de engines activos (default 0.3 = 30%)
    - min_weight_threshold: peso mínimo para considerar engine "activo" (default 0.05)
    - suppression_threshold: supresión máxima aceptable por engine (default 0.8)
    - max_suppressed_ratio: fracción máxima de engines suprimidos (default 0.5)
    
    Uso:
        watchdog = EnsembleWatchdog()
        snapshot = watchdog.evaluate(weights={"taylor": 0.3, "statistical": 0.01},
                                     suppressions={"taylor": 0.1, "statistical": 0.95})
        if snapshot.health == EnsembleHealth.COLLAPSED:
            recovery_manager.force_recovery(snapshot)
    """
    
    def __init__(
        self,
        min_active_ratio: float = 0.3,
        min_weight_threshold: float = 0.05,
        suppression_threshold: float = 0.8,
        max_suppressed_ratio: float = 0.5,
    ):
        self._min_active_ratio = min_active_ratio
        self._min_weight_threshold = min_weight_threshold
        self._suppression_threshold = suppression_threshold
        self._max_suppressed_ratio = max_suppressed_ratio
        self._logger = logging.getLogger(__name__)
    
    def evaluate(
        self,
        weights: dict[str, float],
        suppressions: dict[str, float]
    ) -> WatchdogSnapshot:
        """
        Evalúa salud del ensemble dado pesos y supresiones actuales.
        Retorna snapshot con health status y recomendación.
        
        HEALTHY: active_ratio > 0.5
        DEGRADED: 0.3 <= active_ratio <= 0.5
        CRITICAL: 0 < active_ratio < 0.3
        COLLAPSED: active_ratio == 0 OR all weights <= min_weight_threshold
        """
        if not weights:
            return WatchdogSnapshot(
                total_engines=0,
                active_engines=0,
                suppressed_engines=0,
                min_weight=0.0,
                max_suppression=0.0,
                health=EnsembleHealth.COLLAPSED,
                recovery_recommended=True,
                reason="Empty ensemble",
            )
        
        total_engines = len(weights)
        active_engines = sum(1 for w in weights.values() if w > self._min_weight_threshold)
        suppressed_engines = sum(
            1 for s in suppressions.values() if s > self._suppression_threshold
        )
        min_weight = min(weights.values()) if weights else 0.0
        max_suppression = max(suppressions.values()) if suppressions else 0.0
        
        active_ratio = active_engines / total_engines if total_engines > 0 else 0.0
        suppressed_ratio = suppressed_engines / total_engines if total_engines > 0 else 0.0
        
        # Determine health status
        if active_ratio == 0.0:
            health = EnsembleHealth.COLLAPSED
            reason = "No active engines"
        elif active_ratio < self._min_active_ratio:
            health = EnsembleHealth.CRITICAL
            reason = f"Active ratio {active_ratio:.2f} below threshold {self._min_active_ratio}"
        elif active_ratio < 0.5:
            health = EnsembleHealth.DEGRADED
            reason = f"Active ratio {active_ratio:.2f} in degraded range"
        else:
            health = EnsembleHealth.HEALTHY
            reason = f"Active ratio {active_ratio:.2f} healthy"
        
        recovery_recommended = health in (EnsembleHealth.CRITICAL, EnsembleHealth.COLLAPSED)
        
        return WatchdogSnapshot(
            total_engines=total_engines,
            active_engines=active_engines,
            suppressed_engines=suppressed_engines,
            min_weight=min_weight,
            max_suppression=max_suppression,
            health=health,
            recovery_recommended=recovery_recommended,
            reason=reason,
        )
    
    def should_trigger_recovery(self, snapshot: WatchdogSnapshot) -> bool:
        """Retorna True si se debe activar recovery forzado."""
        return snapshot.health in (EnsembleHealth.CRITICAL, EnsembleHealth.COLLAPSED)
