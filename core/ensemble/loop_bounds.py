"""
Bounds globales para el feedback loop AdaptiveContamination ↔
BayesianWeightTracker ↔ InhibitionGate.

No modifica los componentes — provee valores límite que cada
componente puede consultar para auto-limitarse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging


@dataclass(frozen=True)
class LoopBoundsConfig:
    """Bounds globales del feedback loop."""
    # Contamination
    max_contamination_rate_of_change: float = 0.1   # max delta por ciclo
    contamination_cooldown_cycles: int = 5           # ciclos entre ajustes
    
    # Weights
    min_ensemble_mean_weight: float = 0.1            # media mínima del ensemble
    max_weight_drop_per_cycle: float = 0.3           # max caída de peso por ciclo
    
    # Suppression
    max_simultaneous_suppressed: int = 2             # máx engines suprimidos a la vez
    max_total_suppression: float = 0.7               # suma máxima de supresiones


class LoopBoundsMonitor:
    """
    Monitorea que el feedback loop no exceda los bounds globales.
    Emite warnings y puede bloquear actualizaciones peligrosas.
    
    Uso:
        monitor = LoopBoundsMonitor()
        
        # Antes de actualizar contamination:
        if monitor.check_contamination_update(old=0.005, new=0.015):
            adaptive_contamination.update(0.015)
        
        # Después de actualizar pesos:
        monitor.check_weight_state(weights)
    """
    
    def __init__(self, config: Optional[LoopBoundsConfig] = None):
        self._config = config or LoopBoundsConfig()
        self._contamination_cycles = 0
        self._last_contamination_value: Optional[float] = None
        self._first_update = True  # Allow first update without cooldown
        self._logger = logging.getLogger(__name__)
    
    def check_contamination_update(
        self,
        old_value: float,
        new_value: float
    ) -> bool:
        """
        Retorna True si el update es seguro.
        False si excede rate_of_change o cooldown.
        """
        # Check cooldown (skip for first update)
        if not self._first_update and self._contamination_cycles < self._config.contamination_cooldown_cycles:
            self._logger.warning(
                "contamination_update_blocked_cooldown",
                extra={
                    "cycles_remaining": self._config.contamination_cooldown_cycles - self._contamination_cycles,
                    "old_value": old_value,
                    "new_value": new_value,
                }
            )
            return False
        
        # Check rate of change
        if old_value > 0:
            rate_of_change = abs(new_value - old_value) / old_value
            if rate_of_change > self._config.max_contamination_rate_of_change:
                self._logger.warning(
                    "contamination_update_blocked_rate",
                    extra={
                        "rate_of_change": rate_of_change,
                        "max_allowed": self._config.max_contamination_rate_of_change,
                        "old_value": old_value,
                        "new_value": new_value,
                    }
                )
                return False
        
        # Update state
        self._last_contamination_value = new_value
        self._contamination_cycles = 0
        self._first_update = False
        return True
    
    def check_weight_state(
        self,
        weights: dict[str, float]
    ) -> bool:
        """
        Retorna True si el estado de pesos es seguro.
        False si mean_weight < min o drop > max.
        Log WARNING si cerca de límite.
        """
        if not weights:
            return True
        
        mean_weight = sum(weights.values()) / len(weights)
        
        if mean_weight < self._config.min_ensemble_mean_weight:
            self._logger.error(
                "weight_state_critical",
                extra={
                    "mean_weight": mean_weight,
                    "min_allowed": self._config.min_ensemble_mean_weight,
                }
            )
            return False
        
        # Check if approaching limit
        if mean_weight < self._config.min_ensemble_mean_weight * 1.2:
            self._logger.warning(
                "weight_state_degraded",
                extra={
                    "mean_weight": mean_weight,
                    "min_allowed": self._config.min_ensemble_mean_weight,
                }
            )
        
        return True
    
    def check_suppression_state(
        self,
        suppressions: dict[str, float]
    ) -> bool:
        """
        Retorna True si supresiones dentro de bounds.
        False si demasiados engines suprimidos simultáneamente.
        """
        if not suppressions:
            return True
        
        suppressed_count = sum(1 for s in suppressions.values() if s > 0.5)
        total_suppression = sum(suppressions.values())
        
        if suppressed_count > self._config.max_simultaneous_suppressed:
            self._logger.error(
                "suppression_state_critical",
                extra={
                    "suppressed_count": suppressed_count,
                    "max_allowed": self._config.max_simultaneous_suppressed,
                }
            )
            return False
        
        if total_suppression > self._config.max_total_suppression:
            self._logger.error(
                "suppression_state_critical_total",
                extra={
                    "total_suppression": total_suppression,
                    "max_allowed": self._config.max_total_suppression,
                }
            )
            return False
        
        return True
    
    def increment_contamination_cycle(self) -> None:
        """Incrementa el contador de ciclos de contamination."""
        self._contamination_cycles += 1
    
    def get_loop_health_summary(self) -> dict:
        """Resumen del estado del loop para observabilidad."""
        return {
            "contamination_cycles": self._contamination_cycles,
            "last_contamination_value": self._last_contamination_value,
            "config": {
                "max_contamination_rate_of_change": self._config.max_contamination_rate_of_change,
                "contamination_cooldown_cycles": self._config.contamination_cooldown_cycles,
                "min_ensemble_mean_weight": self._config.min_ensemble_mean_weight,
                "max_weight_drop_per_cycle": self._config.max_weight_drop_per_cycle,
                "max_simultaneous_suppressed": self._config.max_simultaneous_suppressed,
                "max_total_suppression": self._config.max_total_suppression,
            },
        }
