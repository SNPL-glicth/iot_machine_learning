"""DocumentAnalyzer — Adapter para AnalyzeDocumentUseCase.

Este archivo mantiene backward compatibility mientras delega
100% del comportamiento a AnalyzeDocumentUseCase.

ESTADO: En eliminación progresiva.
- No agregar nuevas features aquí
- Migrar callers a application.analyze_document.AnalyzeDocumentUseCase
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from iot_machine_learning.application.analyze_document import (
    AnalyzeDocumentUseCase,
)
from iot_machine_learning.domain.ports.analysis import (
    AnalysisEnginePort,
)
from iot_machine_learning.domain.ports.document_analysis import (
    CachePort,
    DocumentPersistencePort,
    PlasticityPort,
)


class DocumentAnalyzer:
    """Adapter legacy — delega 100% a AnalyzeDocumentUseCase.
    
    DEPRECADO: Usar directamente application.analyze_document.AnalyzeDocumentUseCase
    """
    
    def __init__(
        self,
        engine: AnalysisEnginePort,
        cache: Optional[CachePort] = None,
        persistence: Optional[DocumentPersistencePort] = None,
        plasticity: Optional[PlasticityPort] = None,
    ):
        """Initialize adapter with injected dependencies.
        
        Args:
            engine: Motor de análisis (requerido)
            cache: Puerto de cache (opcional)
            persistence: Puerto de persistencia (opcional)
            plasticity: Puerto de plasticidad (opcional)
        """
        warnings.warn(
            "DocumentAnalyzer está deprecado. Usa AnalyzeDocumentUseCase.",
            DeprecationWarning,
            stacklevel=2,
        )
        
        # Use case con DI pura — ninguna creación interna
        self._use_case = AnalyzeDocumentUseCase(
            engine=engine,
            cache=cache,
            persistence=persistence,
            plasticity=plasticity,
        )
    
    def analyze(
        self,
        document_id: str,
        content_type: str,
        normalized_payload: Dict[str, Any],
        tenant_id: str = "",
    ) -> Dict[str, Any]:
        """Analyze document — delega a AnalyzeDocumentUseCase.
        
        Args:
            document_id: ID único del documento
            content_type: Tipo de contenido
            normalized_payload: Payload normalizado (legacy)
            tenant_id: ID del tenant
            
        Returns:
            Dict compatible con código legacy
        """
        # Extraer contenido del payload legacy
        content = self._extract_content(normalized_payload, content_type)
        
        # Delegar al use case
        output = self._use_case.execute(
            document_id=document_id,
            content=content,
            tenant_id=tenant_id,
            content_type=content_type,
        )
        
        # Convertir AnalysisOutput a dict legacy
        return self._output_to_dict(output)
    
    def _extract_content(self, payload: Dict[str, Any], content_type: str) -> str:
        """Extrae contenido string del payload legacy."""
        # Extraer de data.full_text o data directamente
        if "data" in payload:
            data = payload["data"]
            if isinstance(data, dict):
                if "full_text" in data:
                    return str(data["full_text"])
                elif "text" in data:
                    return str(data["text"])
                elif "content" in data:
                    return str(data["content"])
            elif isinstance(data, str):
                return data
        
        # Fallback: convertir todo el payload a string
        return str(payload)
    
    def _output_to_dict(self, output: Any) -> Dict[str, Any]:
        """Convierte AnalysisOutput a dict legacy."""
        if hasattr(output, "to_dict"):
            return output.to_dict()
        
        return {
            "document_id": output.document_id,
            "tenant_id": output.tenant_id,
            "classification": output.classification,
            "conclusion": output.conclusion,
            "confidence": output.confidence,
            "analysis": output.analysis,
            "explanation": output.explanation,
            "comparative_context": output.comparative_context,
            "processing_time_ms": output.processing_time_ms,
            "cached": output.cached,
        }


# Mantener alias para imports directos
__all__ = ["DocumentAnalyzer"]
