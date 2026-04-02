"""Pipeline Executor — MED-1 Refactored.

Orquestador de fases del pipeline cognitivo usando patrón Strategy.
Cada fase es independiente y el executor solo coordina la ejecución.

Reducción: 666 líneas → ~180 líneas (73% de reducción)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from ...interfaces import PredictionResult

from .phases import PipelineContext, create_initial_context
from .phases.boundary_check_phase import BoundaryCheckPhase
from .phases.perceive_phase import PerceivePhase
from .phases.predict_phase import PredictPhase
from .phases.adapt_phase import AdaptPhase
from .phases.inhibit_phase import InhibitPhase
from .phases.fuse_phase import FusePhase
from .phases.decision_arbiter_phase import DecisionArbiterPhase
from .phases.coherence_check_phase import CoherenceCheckPhase
from .phases.confidence_calibration_phase import ConfidenceCalibrationPhase
from .phases.explain_phase import ExplainPhase
from .phases.action_guard_phase import ActionGuardPhase
from .phases.narrative_unification_phase import NarrativeUnificationPhase
from .phases.assembly_phase import AssemblyPhase
from ..analysis.types import PipelineTimer

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Ejecutor del pipeline cognitivo.
    
    Responsabilidad única: orquestar la ejecución secuencial de fases.
    La lógica pesada está delegada a las clases de fase individuales.
    
    Attributes:
        phases: Lista ordenada de fases a ejecutar
    """
    
    def __init__(self) -> None:
        """Inicializa el executor con todas las fases configuradas."""
        self._phases = [
            BoundaryCheckPhase(),
            PerceivePhase(),
            PredictPhase(),
            AdaptPhase(),
            InhibitPhase(),
            FusePhase(),
            DecisionArbiterPhase(),
            CoherenceCheckPhase(),
            ConfidenceCalibrationPhase(),
            ExplainPhase(),
            ActionGuardPhase(),
            NarrativeUnificationPhase(),
        ]
        self._assembly = AssemblyPhase()
    
    def execute(
        self,
        orchestrator,
        values: List[float],
        timestamps: Optional[List[float]],
        series_id: str,
        flags: Any,
    ) -> PredictionResult:
        """Ejecuta el pipeline completo.
        
        Args:
            orchestrator: MetaCognitiveOrchestrator instance
            values: Time series values
            timestamps: Optional timestamps
            series_id: Series identifier
            flags: Feature flags snapshot (required)
        
        Returns:
            PredictionResult with cognitive metadata
        """
        # Validar flags (CRIT-3)
        if flags is None:
            raise ValueError(
                "flags is required. Pass feature flags from orchestrator "
                "to enable testable dependency injection."
            )
        
        # Crear contexto inicial
        timer = PipelineTimer(budget_ms=orchestrator._budget_ms)
        ctx = create_initial_context(
            orchestrator=orchestrator,
            values=values,
            timestamps=timestamps,
            series_id=series_id,
            flags=flags,
            timer=timer,
        )
        
        # Ejecutar fases secuencialmente
        for phase in self._phases:
            ctx = phase.execute(ctx)
            
            # Early termination si es out-of-domain
            if ctx.is_fallback and ctx.fallback_reason == "out_of_domain":
                return self._create_early_result(ctx)
            
            # Early termination si es fallback normal
            if ctx.is_fallback and ctx.diagnostic is not None:
                return self._create_fallback_result(ctx)
        
        # Fase final: ensamblar resultado
        return self._assembly.execute(ctx)
    
    def _create_early_result(self, ctx: PipelineContext) -> PredictionResult:
        """Crea resultado temprano para out-of-domain."""
        phase = BoundaryCheckPhase()
        return phase.create_early_result(ctx)
    
    def _create_fallback_result(self, ctx: PipelineContext) -> PredictionResult:
        """Crea resultado para fallback."""
        from ...interfaces import PredictionResult
        
        return PredictionResult(
            predicted_value=ctx.orchestrator._last_explanation.predicted_value 
                if ctx.orchestrator._last_explanation else None,
            confidence=0.2,
            trend="unknown",
            metadata=ctx.metadata,
        )


# Instancia global del executor (singleton pattern)
_pipeline_executor = PipelineExecutor()


def execute_pipeline(
    orchestrator,
    values: List[float],
    timestamps: Optional[List[float]],
    series_id: str,
    flags_snapshot: Optional[Any] = None,
) -> PredictionResult:
    """Entry point para ejecutar el pipeline.
    
    Args:
        orchestrator: MetaCognitiveOrchestrator instance
        values: Time series values
        timestamps: Optional timestamps
        series_id: Series identifier
        flags_snapshot: Feature flags snapshot (required por CRIT-3)
    
    Returns:
        PredictionResult with cognitive metadata
    """
    return _pipeline_executor.execute(
        orchestrator=orchestrator,
        values=values,
        timestamps=timestamps,
        series_id=series_id,
        flags=flags_snapshot,
    )
