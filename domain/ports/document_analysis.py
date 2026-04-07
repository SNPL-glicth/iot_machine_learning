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
    ) -> "AnalysisOutput":
        """Construye desde AnalysisResult del motor unificado."""
        return cls(
            document_id=document_id,
            tenant_id=tenant_id,
            classification=result.signal.input_type.value,
            conclusion=result.explanation.narrative,
            confidence=result.decision.confidence,
            analysis={
                "domain": result.signal.domain,
                "severity": result.decision.severity,
                "features": result.signal.features,
                "perceptions": len(result.decision.perceptions),
            },
            explanation=result.explanation,
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
