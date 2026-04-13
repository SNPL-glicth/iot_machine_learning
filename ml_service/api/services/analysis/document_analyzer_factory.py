"""Factory for creating DocumentAnalyzer with dependency injection.

Encapsula la construcción de todas las dependencias necesarias
para DocumentAnalyzer, manteniendo el caller (ZeninQueuePoller) limpio.
"""

from __future__ import annotations

from typing import Optional, Any

from iot_machine_learning.domain.ports.analysis import AnalysisEnginePort
from iot_machine_learning.domain.ports.document_analysis import (
    CachePort,
    DocumentPersistencePort,
    PlasticityPort,
)

from .cache import AnalysisCache
from .arbitrator import EngineArbitrator
from .feedback_loop import PlasticityFeedbackLoop
from .decision_engine_service import DecisionEngineService
from .decision_context_builder import DecisionContextBuilder
from ..document_analyzer import DocumentAnalyzer


def create_document_analyzer(
    feature_flags: Optional[Any] = None,
    engine: Optional[AnalysisEnginePort] = None,
    cache: Optional[CachePort] = None,
    persistence: Optional[DocumentPersistencePort] = None,
    plasticity: Optional[PlasticityPort] = None,
) -> DocumentAnalyzer:
    """Factory function para crear DocumentAnalyzer con DI.
    
    Args:
        feature_flags: Optional feature flags para configuración
        engine: Optional engine custom (si no, se crea default)
        cache: Optional cache custom (si no, se crea AnalysisCache)
        persistence: Optional persistence port
        plasticity: Optional plasticity port
        
    Returns:
        DocumentAnalyzer configurado con todas las dependencias
    """
    # Crear cache default si no viene
    if cache is None:
        cache = AnalysisCache(max_entries=100)
    
    # Crear engine default si no viene
    if engine is None:
        engine = _create_default_engine(
            feature_flags=feature_flags,
        )
    
    # Crear DocumentAnalyzer con DI pura
    return DocumentAnalyzer(
        engine=engine,
        cache=cache,
        persistence=persistence,
        plasticity=plasticity,
    )


def _create_default_engine(feature_flags: Optional[Any] = None) -> AnalysisEnginePort:
    """Crea el engine por defecto con todas sus dependencias.
    
    Esta es la lógica que antes estaba dentro de DocumentAnalyzer.__init__
    """
    from iot_machine_learning.infrastructure.ml.cognitive.universal import (
        UniversalAnalysisEngine,
        UniversalComparativeEngine,
    )
    
    # Extract deterministic mode from feature flags
    deterministic_mode = False
    analysis_seed = 42
    if feature_flags is not None:
        if hasattr(feature_flags, 'cognitive'):
            cognitive = feature_flags.cognitive
            deterministic_mode = getattr(cognitive, 'ZENIN_DETERMINISTIC_MODE', False)
            analysis_seed = getattr(cognitive, 'ZENIN_ANALYSIS_SEED', 42)
    
    # Inicializar engines universales si están disponibles
    analysis_engine = None
    comparative_engine = None
    try:
        analysis_engine = UniversalAnalysisEngine(
            deterministic_mode=deterministic_mode,
            analysis_seed=analysis_seed,
        )
        comparative_engine = UniversalComparativeEngine()
    except Exception:
        pass  # Graceful fallback
    
    # Crear componentes adicionales
    arbitrator = EngineArbitrator()
    feedback_loop = PlasticityFeedbackLoop(
        deterministic_mode=deterministic_mode,
    )
    decision_service = DecisionEngineService(
        feature_flags=feature_flags,
        decision_engine=None,
    )
    context_builder = DecisionContextBuilder(
        anomaly_tracker=None,
    )
    
    # Retornar engine adapter con todas las dependencias
    return _EngineAdapter(
        analysis_engine=analysis_engine,
        comparative_engine=comparative_engine,
        arbitrator=arbitrator,
        feedback_loop=feedback_loop,
        decision_service=decision_service,
        context_builder=context_builder,
    )


class _EngineAdapter:
    """Adapter interno que usa los componentes extraídos como engine."""
    
    def __init__(
        self,
        analysis_engine: Any,
        comparative_engine: Any,
        arbitrator: Any,
        feedback_loop: Any,
        decision_service: Any,
        context_builder: Any,
    ) -> None:
        self._analysis_engine = analysis_engine
        self._comparative_engine = comparative_engine
        self._arbitrator = arbitrator
        self._feedback_loop = feedback_loop
        self._decision_service = decision_service
        self._context_builder = context_builder
    
    def analyze(self, content: str, context: Any) -> Any:
        """Ejecuta análisis usando los componentes."""
        from iot_machine_learning.ml_service.api.services.analysis import analyze_with_universal
        from iot_machine_learning.domain.ports.analysis import AnalysisResult, Signal, Decision, Explanation
        
        payload = {"data": {"full_text": content}}
        
        # Determinar content_type
        if context.input_type is None:
            content_type = "text"
        elif hasattr(context.input_type, 'value'):
            content_type = context.input_type.value
        else:
            content_type = str(context.input_type)
        
        # Si no hay engines disponibles, retornar resultado básico
        if self._analysis_engine is None:
            return self._create_basic_result(content, context)
        
        # Llamar al análisis universal
        universal_result, _, semantic_conclusion = analyze_with_universal(
            document_id=context.series_id,
            content_type=content_type,
            payload=payload,
            tenant_id=context.tenant_id,
            analysis_engine=self._analysis_engine,
            comparative_engine=self._comparative_engine,
            cognitive_memory=None,
        )
        
        # Convertir resultado a AnalysisResult (pasar semantic_conclusion)
        return self._convert_to_analysis_result(universal_result, semantic_conclusion)
    
    def _create_basic_result(self, content: str, context: Any) -> Any:
        """Crea resultado básico cuando no hay engines."""
        from iot_machine_learning.domain.ports.analysis import Signal, Decision, Explanation, AnalysisResult
        
        signal = Signal(
            raw_data=content,
            input_type=context.input_type,
            domain="general",
            features={},
        )
        decision = Decision(
            severity="info",
            confidence=0.5,
            perceptions=[],
            weights={},
        )
        explanation = Explanation(
            narrative="Analysis engine not available",
            confidence=0.5,
            contributions={},
            reasoning_trace=[],
        )
        
        return AnalysisResult(
            signal=signal,
            decision=decision,
            explanation=explanation,
        )
    
    def _convert_to_analysis_result(self, universal_result: Any, semantic_conclusion: Optional[str] = None) -> Any:
        """Convierte resultado universal a AnalysisResult."""
        from iot_machine_learning.domain.ports.analysis import Signal, Decision, Explanation, AnalysisResult
        
        domain = getattr(universal_result, 'domain', 'general')
        confidence = getattr(universal_result, 'confidence', 0.5)
        
        # Extract severity from UniversalResult
        severity = "info"
        if hasattr(universal_result, 'severity') and universal_result.severity:
            severity_obj = universal_result.severity
            if hasattr(severity_obj, 'severity'):
                severity = severity_obj.severity
            else:
                severity = str(severity_obj)
        
        # Use semantic_conclusion if provided, otherwise fallback
        narrative = semantic_conclusion if semantic_conclusion else "Analysis completed"
        
        signal = Signal(
            raw_data="",
            input_type=None,
            domain=domain,
            features={},
        )
        decision = Decision(
            severity=severity,
            confidence=confidence,
            perceptions=[],
            weights={},
            selection_reason=narrative[:100] if narrative else "",
        )
        explanation = Explanation(
            narrative=narrative,
            confidence=confidence,
            contributions={},
            reasoning_trace=[],
            domain=domain,
            severity=severity,
        )
        
        return AnalysisResult(
            signal=signal,
            decision=decision,
            explanation=explanation,
        )
