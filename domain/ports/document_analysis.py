"""Puertos para análisis de documentos."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class AnalysisOutput:
    """Output de análisis de documento."""
    document_id: str
    tenant_id: str
    classification: str
    conclusion: str
    confidence: float
    analysis: Dict[str, Any]
    explanation: Optional[Any] = None  # Explanation domain object
    comparative_context: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0.0
    cached: bool = False
    
    @classmethod
    def from_result(
        cls,
        result: Any,
        document_id: str,
        tenant_id: str,
        semantic_conclusion: Optional[str] = None,
    ) -> "AnalysisOutput":
        """Construye desde AnalysisResult del motor unificado."""
        import logging
        logger = logging.getLogger(__name__)
        
        # DEBUG: Log what we're receiving
        logger.warning(
            f"[FROM_RESULT] Building output from result type: {type(result).__name__}, has_semantic={semantic_conclusion is not None}",
            extra={
                "document_id": document_id,
                "has_domain": hasattr(result, 'domain'),
                "has_confidence": hasattr(result, 'confidence'),
                "has_severity": hasattr(result, 'severity'),
                "has_analysis": hasattr(result, 'analysis'),
                "has_explanation": hasattr(result, 'explanation'),
            }
        )
        
        # Safe navigation for input_type (can be None)
        input_type_value = "unknown"
        if hasattr(result, 'signal') and result.signal is not None:
            if hasattr(result.signal, 'input_type') and result.signal.input_type is not None:
                input_type_value = result.signal.input_type.value if hasattr(result.signal.input_type, 'value') else str(result.signal.input_type)
        elif hasattr(result, 'input_type') and result.input_type is not None:
            input_type_value = result.input_type.value if hasattr(result.input_type, 'value') else str(result.input_type)
        
        # Safe navigation for other fields - extract BEFORE format_conclusion
        domain = "unknown"
        if hasattr(result, 'signal') and result.signal is not None and hasattr(result.signal, 'domain'):
            domain = result.signal.domain
        elif hasattr(result, 'domain'):
            domain = result.domain
        
        confidence = 0.5
        if hasattr(result, 'decision') and result.decision is not None and hasattr(result.decision, 'confidence'):
            confidence = result.decision.confidence
        elif hasattr(result, 'confidence'):
            confidence = result.confidence
        
        severity = "info"
        if hasattr(result, 'decision') and result.decision is not None and hasattr(result.decision, 'severity'):
            severity = result.decision.severity
        elif hasattr(result, 'severity') and result.severity is not None:
            severity = result.severity.severity if hasattr(result.severity, 'severity') else str(result.severity)
        
        # DEBUG: Log extracted values
        logger.warning(
            f"[FROM_RESULT] Extracted fields: domain={domain}, confidence={confidence}, severity={severity}"
        )
        
        # Use format_conclusion for rich output
        narrative = "Analysis completed"
        try:
            # If semantic_conclusion is provided, use it directly
            if semantic_conclusion and len(semantic_conclusion) > 50:
                logger.warning(f"[FROM_RESULT] Using provided semantic_conclusion: {semantic_conclusion[:100]}...")
                narrative = semantic_conclusion
            # Check if this is AnalysisResult wrapper (data is in nested structure)
            elif hasattr(result, 'signal') and hasattr(result, 'decision') and hasattr(result, 'explanation'):
                # AnalysisResult wrapper - build narrative manually since data doesn't transfer
                logger.warning("[FROM_RESULT] AnalysisResult detected - building narrative manually")
                
                # Use explanation.narrative if it exists and is not generic
                if result.explanation and hasattr(result.explanation, 'narrative'):
                    narrative = result.explanation.narrative
                    logger.warning(f"[FROM_RESULT] explanation.narrative = '{narrative[:200] if narrative else 'NONE'}'")
                    
                    if narrative and narrative != "Analysis completed" and len(narrative) > 50:
                        logger.warning(f"[FROM_RESULT] Using rich explanation.narrative")
                    else:
                        # Build from available fields
                        narrative = f"{domain.title()} incident — {severity.title()} | Confidence: {confidence:.1%}"
                        logger.warning(f"[FROM_RESULT] explanation.narrative too short/generic - built from fields: {narrative}")
                else:
                    # Build from available fields
                    narrative = f"{domain.title()} incident — {severity.title()} | Confidence: {confidence:.1%}"
                    logger.warning(f"[FROM_RESULT] Built narrative from fields: {narrative}")
            else:
                # UniversalResult - call format_conclusion
                from iot_machine_learning.ml_service.api.services.analysis.conclusion_formatter import format_conclusion
                logger.warning("[FROM_RESULT] UniversalResult detected - calling format_conclusion")
                narrative = format_conclusion(result)
        except Exception as exc:
            # Log the error for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"format_conclusion_failed: {exc}",
                exc_info=True,
                extra={
                    "has_explanation": hasattr(result, 'explanation'),
                    "has_domain": hasattr(result, 'domain'),
                    "has_severity": hasattr(result, 'severity'),
                    "has_confidence": hasattr(result, 'confidence'),
                }
            )
            # Fallback to explanation narrative
            if hasattr(result, 'explanation') and result.explanation is not None and hasattr(result.explanation, 'narrative'):
                narrative = result.explanation.narrative
            else:
                # Last resort: build simple narrative from available fields
                try:
                    domain_str = domain if domain != "unknown" else "Analysis"
                    severity_str = severity if severity != "info" else "completed"
                    narrative = f"{domain_str.title()} — {severity_str.title()} | Confidence: {confidence:.1%}"
                except Exception:
                    narrative = "Analysis completed"
        
        features = {}
        if hasattr(result, 'signal') and result.signal is not None and hasattr(result.signal, 'features'):
            features = result.signal.features
        
        perceptions_count = 0
        if hasattr(result, 'decision') and result.decision is not None and hasattr(result.decision, 'perceptions'):
            perceptions_count = len(result.decision.perceptions) if result.decision.perceptions else 0
        elif hasattr(result, 'analysis') and isinstance(result.analysis, dict):
            perceptions = result.analysis.get('perceptions', [])
            perceptions_count = len(perceptions) if perceptions else 0
        
        explanation_obj = None
        if hasattr(result, 'explanation'):
            explanation_obj = result.explanation
        
        return cls(
            document_id=document_id,
            tenant_id=tenant_id,
            classification=input_type_value,
            conclusion=narrative,
            confidence=confidence,
            analysis={
                "domain": domain,
                "severity": severity,
                "features": features,
                "perceptions": perceptions_count,
            },
            explanation=explanation_obj,
        )


class CachePort(ABC):
    """Puerto para cache de análisis."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[AnalysisOutput]:
        """Obtiene resultado cacheado."""
        ...
    
    @abstractmethod
    def set(self, key: str, value: AnalysisOutput, ttl: Optional[int] = None) -> None:
        """Guarda resultado en cache con TTL opcional."""
        ...
    
    @abstractmethod
    def invalidate(self, key: str) -> None:
        """Invalida entrada de cache."""
        ...


class DocumentPersistencePort(ABC):
    """Puerto para persistencia de análisis de documentos."""
    
    @abstractmethod
    def save_analysis(self, output: AnalysisOutput) -> None:
        """Guarda resultado de análisis."""
        ...
    
    @abstractmethod
    def get_analysis(self, document_id: str) -> Optional[AnalysisOutput]:
        """Obtiene análisis por document_id."""
        ...


class PlasticityPort(ABC):
    """Puerto para feedback de plasticidad."""
    
    @abstractmethod
    def update(self, domain: str, result: Any) -> None:
        """Actualiza pesos de plasticidad basado en resultado."""
        ...
