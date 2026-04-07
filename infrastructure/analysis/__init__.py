"""Motor de análisis unificado — factory y exports."""

from __future__ import annotations

from typing import Dict, List, Optional

from iot_machine_learning.domain.ports.analysis import (
    AnalysisContext,
    AnalysisEnginePort,
    InputType,
)
from .engine import UnifiedAnalysisEngine
from .pipeline import PerceivePhase, AnalyzePhase, ReasonPhase, ExplainPhase
from .adapters import (
    SimpleTypeDetector,
    SimpleDomainClassifier,
    SimpleFeatureExtractor,
    SimplePerceptionCollector,
    SimpleInhibitor,
    SimpleFusion,
    SimpleSeverityClassifier,
)


def build_unified_engine(
    enable_plasticity: bool = True,
    budget_ms: float = 2000.0,
) -> UnifiedAnalysisEngine:
    """Construye motor unificado con dependencias por defecto.
    
    Factory que ensambla el pipeline completo con implementaciones simples.
    Para producción, inyectar implementaciones reales de los motores existentes.
    
    Args:
        enable_plasticity: Habilitar aprendizaje adaptativo
        budget_ms: Presupuesto de tiempo en milisegundos
    
    Returns:
        UnifiedAnalysisEngine configurado
    """
    # Fase 1: PERCEIVE
    type_detectors = [SimpleTypeDetector()]
    domain_classifier = SimpleDomainClassifier()
    feature_extractor = SimpleFeatureExtractor()
    
    perceive = PerceivePhase(
        type_detectors=type_detectors,
        domain_classifier=domain_classifier,
        feature_extractor=feature_extractor,
    )
    
    # Fase 2: ANALYZE
    collectors = {
        InputType.TEXT: SimplePerceptionCollector("text"),
        InputType.TIMESERIES: SimplePerceptionCollector("timeseries"),
        InputType.DOCUMENT: SimplePerceptionCollector("document"),
        InputType.TABULAR: SimplePerceptionCollector("tabular"),
    }
    
    analyze = AnalyzePhase(collectors=collectors)
    
    # Fase 3: REASON
    inhibitor = SimpleInhibitor()
    plasticity = None  # TODO: Inyectar PlasticityTracker real
    fusion = SimpleFusion()
    severity_classifier = SimpleSeverityClassifier()
    
    reason = ReasonPhase(
        inhibitor=inhibitor,
        plasticity=plasticity,
        fusion=fusion,
        severity_classifier=severity_classifier,
    )
    
    # Fase 4: EXPLAIN
    explain = ExplainPhase(narrative_builder=None)
    
    # Ensamblar motor
    return UnifiedAnalysisEngine(
        perceive=perceive,
        analyze=analyze,
        reason=reason,
        explain=explain,
        budget_ms=budget_ms,
    )


__all__ = [
    "UnifiedAnalysisEngine",
    "build_unified_engine",
    "PerceivePhase",
    "AnalyzePhase",
    "ReasonPhase",
    "ExplainPhase",
    "AnalysisContext",
    "AnalysisEnginePort",
]
