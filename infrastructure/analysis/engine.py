"""Motor de análisis unificado — consolida TextCognitive, Universal y MetaCognitive."""

from __future__ import annotations

import logging
from typing import Any, Dict

from iot_machine_learning.domain.ports.analysis import (
    AnalysisContext,
    AnalysisEnginePort,
    AnalysisResult,
)
from .pipeline.perceive import PerceivePhase
from .pipeline.analyze import AnalyzePhase
from .pipeline.reason import ReasonPhase
from .pipeline.explain import ExplainPhase

logger = logging.getLogger(__name__)


class UnifiedAnalysisEngine(AnalysisEnginePort):
    """Motor unificado de análisis cognitivo.
    
    Única responsabilidad: ejecutar pipeline de 4 fases.
    
    Pipeline:
        1. PERCEIVE — Detecta tipo, clasifica dominio, construye señal
        2. ANALYZE  — Colecta percepciones de analizadores específicos
        3. REASON   — Inhibe, adapta, fusiona percepciones
        4. EXPLAIN  — Construye explicación legible
    
    Args:
        perceive: Fase de percepción (inyectada)
        analyze: Fase de análisis (inyectada)
        reason: Fase de razonamiento (inyectada)
        explain: Fase de explicación (inyectada)
        budget_ms: Presupuesto de tiempo en milisegundos
    """
    
    def __init__(
        self,
        perceive: PerceivePhase,
        analyze: AnalyzePhase,
        reason: ReasonPhase,
        explain: ExplainPhase,
        budget_ms: float = 2000.0,
    ) -> None:
        """Inicializa motor con fases inyectadas."""
        self._perceive = perceive
        self._analyze = analyze
        self._reason = reason
        self._explain = explain
        self._budget_ms = budget_ms
    
    def analyze(
        self,
        raw_data: Any,
        context: AnalysisContext,
    ) -> AnalysisResult:
        """Ejecuta pipeline completo de análisis.
        
        Args:
            raw_data: Dato de entrada (str, List[float], Dict, etc.)
            context: Contexto de análisis con tenant_id, series_id, etc.
        
        Returns:
            AnalysisResult con señal, decisión y explicación
        """
        timing: Dict[str, float] = {}
        
        try:
            # Fase 1: PERCEIVE
            signal = self._perceive.execute(raw_data, context, timing)
            
            # Fase 2: ANALYZE
            perceptions = self._analyze.execute(signal, context, timing)
            
            # Fase 3: REASON
            decision = self._reason.execute(perceptions, signal, context, timing)
            
            # Fase 4: EXPLAIN
            explanation = self._explain.execute(decision, signal, context, timing)
            
            logger.debug(
                "unified_analysis_complete",
                extra={
                    "series_id": context.series_id,
                    "domain": signal.domain,
                    "severity": decision.severity,
                    "confidence": round(decision.confidence, 3),
                    "total_ms": round(sum(timing.values()), 2),
                },
            )
            
            return AnalysisResult(
                signal=signal,
                decision=decision,
                explanation=explanation,
                pipeline_timing=timing,
            )
        
        except Exception as e:
            logger.error(
                "unified_analysis_failed",
                extra={"series_id": context.series_id, "error": str(e)},
                exc_info=True,
            )
            raise
    
    @property
    def name(self) -> str:
        """Nombre del motor."""
        return "unified_analysis_engine"
