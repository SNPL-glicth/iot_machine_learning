"""
Recovery forzado para ensemble collapse.
Restaura engines inhibidos a estado mínimo funcional.

Estrategias de recovery:
- SOFT: reducir supresión 50% en engines más inhibidos
- HARD: resetear supresión a 0 en todos los engines (emergencia)
- SELECTIVE: restaurar solo los N engines menos erróneos
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

from .ensemble_watchdog import EnsembleHealth, WatchdogSnapshot


class RecoveryStrategy(Enum):
    """Estrategia de recovery."""
    SOFT = "soft"           # Reduce supresión 50%, preserva historial
    HARD = "hard"           # Reset completo de supresión (emergencia)
    SELECTIVE = "selective" # Restaura los N engines más confiables


@dataclass
class RecoveryResult:
    """Resultado de una operación de recovery."""
    strategy_used: RecoveryStrategy
    engines_recovered: list[str]
    weight_adjustments: dict[str, float]  # engine -> nuevo peso sugerido
    reason: str
    success: bool


class ForcedRecoveryManager:
    """
    Ejecuta recovery forzado cuando EnsembleWatchdog detecta collapse.
    
    NO accede directamente a InhibitionGate ni BayesianWeightTracker.
    Retorna RecoveryResult con ajustes sugeridos que el caller aplica.
    
    Esto mantiene SRP: recovery calcula QUÉ cambiar,
    el caller decide CÓMO aplicarlo.
    
    Uso:
        recovery = ForcedRecoveryManager()
        result = recovery.execute(
            snapshot=watchdog_snapshot,
            current_weights={"taylor": 0.01, "statistical": 0.01},
            current_suppressions={"taylor": 0.95, "statistical": 0.90},
            engine_errors={"taylor": 0.3, "statistical": 0.5}
        )
        # Aplicar result.weight_adjustments en WeightedFusion
    """
    
    def __init__(
        self,
        soft_reduction_factor: float = 0.5,   # reduce supresión 50%
        min_recovery_weight: float = 0.1,      # peso mínimo post-recovery
        max_engines_to_recover: int = 2,        # SELECTIVE: máx engines a restaurar
    ):
        self._soft_reduction_factor = soft_reduction_factor
        self._min_recovery_weight = min_recovery_weight
        self._max_engines_to_recover = max_engines_to_recover
        self._logger = logging.getLogger(__name__)
    
    def execute(
        self,
        snapshot: WatchdogSnapshot,
        current_weights: dict[str, float],
        current_suppressions: dict[str, float],
        engine_errors: Optional[dict[str, float]] = None
    ) -> RecoveryResult:
        """
        Ejecuta recovery basado en health status del snapshot.
        - COLLAPSED → HARD recovery
        - CRITICAL → SOFT o SELECTIVE basado en engine_errors
        - DEGRADED → SELECTIVE (solo los más inhibidos)
        """
        if snapshot.health == EnsembleHealth.COLLAPSED:
            return self._hard_recovery(current_weights, current_suppressions)
        elif snapshot.health == EnsembleHealth.CRITICAL:
            if engine_errors:
                return self._selective_recovery(
                    current_weights, current_suppressions, engine_errors
                )
            else:
                return self._soft_recovery(current_weights, current_suppressions)
        elif snapshot.health == EnsembleHealth.DEGRADED:
            return self._selective_recovery(
                current_weights, current_suppressions, engine_errors
            )
        else:
            return RecoveryResult(
                strategy_used=RecoveryStrategy.SOFT,
                engines_recovered=[],
                weight_adjustments={},
                reason="No recovery needed - ensemble healthy",
                success=False,
            )
    
    def _soft_recovery(
        self,
        current_weights: dict[str, float],
        current_suppressions: dict[str, float]
    ) -> RecoveryResult:
        """Reduce supresión 50% en todos los engines."""
        weight_adjustments = {}
        engines_recovered = []
        
        for engine, weight in current_weights.items():
            if weight < self._min_recovery_weight:
                new_weight = max(self._min_recovery_weight, weight * 2.0)
                weight_adjustments[engine] = new_weight
                engines_recovered.append(engine)
                self._logger.info(
                    "soft_recovery_applied",
                    extra={"engine": engine, "old_weight": weight, "new_weight": new_weight}
                )
        
        return RecoveryResult(
            strategy_used=RecoveryStrategy.SOFT,
            engines_recovered=engines_recovered,
            weight_adjustments=weight_adjustments,
            reason=f"Soft recovery: doubled weights below {self._min_recovery_weight}",
            success=len(engines_recovered) > 0,
        )
    
    def _hard_recovery(
        self,
        current_weights: dict[str, float],
        current_suppressions: dict[str, float]
    ) -> RecoveryResult:
        """Reset completo — emergencia total."""
        weight_adjustments = {}
        engines_recovered = []
        
        # Reset all weights to equal distribution
        n_engines = len(current_weights)
        if n_engines > 0:
            equal_weight = 1.0 / n_engines
            for engine in current_weights:
                weight_adjustments[engine] = equal_weight
                engines_recovered.append(engine)
        
        self._logger.warning(
            "hard_recovery_executed",
            extra={"n_engines": n_engines, "equal_weight": 1.0 / n_engines if n_engines > 0 else 0.0}
        )
        
        return RecoveryResult(
            strategy_used=RecoveryStrategy.HARD,
            engines_recovered=engines_recovered,
            weight_adjustments=weight_adjustments,
            reason="Hard recovery: reset all weights to equal distribution",
            success=len(engines_recovered) > 0,
        )
    
    def _selective_recovery(
        self,
        current_weights: dict[str, float],
        current_suppressions: dict[str, float],
        engine_errors: Optional[dict[str, float]] = None
    ) -> RecoveryResult:
        """Restaura los N engines con menor error histórico."""
        weight_adjustments = {}
        engines_recovered = []
        
        if not engine_errors:
            # Fallback: recover engines with lowest suppression
            sorted_by_suppression = sorted(
                current_suppressions.items(), key=lambda x: x[1]
            )
            for engine, _ in sorted_by_suppression[:self._max_engines_to_recover]:
                if current_weights.get(engine, 0.0) < self._min_recovery_weight:
                    weight_adjustments[engine] = self._min_recovery_weight
                    engines_recovered.append(engine)
        else:
            # Recover engines with lowest error
            sorted_by_error = sorted(engine_errors.items(), key=lambda x: x[1])
            for engine, _ in sorted_by_error[:self._max_engines_to_recover]:
                if current_weights.get(engine, 0.0) < self._min_recovery_weight:
                    weight_adjustments[engine] = self._min_recovery_weight
                    engines_recovered.append(engine)
                    self._logger.info(
                        "selective_recovery_applied",
                        extra={"engine": engine, "error": engine_errors[engine]}
                    )
        
        return RecoveryResult(
            strategy_used=RecoveryStrategy.SELECTIVE,
            engines_recovered=engines_recovered,
            weight_adjustments=weight_adjustments,
            reason=f"Selective recovery: restored {len(engines_recovered)} best engines",
            success=len(engines_recovered) > 0,
        )
