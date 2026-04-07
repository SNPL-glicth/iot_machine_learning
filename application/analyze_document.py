"""Caso de uso: Analizar documento con cache y persistencia."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from iot_machine_learning.domain.ports.analysis import (
    AnalysisContext,
    AnalysisEnginePort,
    InputType,
)
from iot_machine_learning.domain.ports.document_analysis import (
    AnalysisOutput,
    CachePort,
    DocumentPersistencePort,
    PlasticityPort,
)
from iot_machine_learning.domain.services.plasticity_feedback import (
    update_plasticity_from_result,
)
from iot_machine_learning.infrastructure.persistence.cache import (
    compute_content_hash,
    build_cache_key,
)

logger = logging.getLogger(__name__)


class AnalyzeDocumentUseCase:
    """Caso de uso: orquesta análisis de documento con cache y persistencia.
    
    Responsabilidad única: coordinar cache → engine → persist → feedback.
    Sin lógica de ML. Sin imports condicionales. Sin estado global.
    
    Args:
        engine: Motor de análisis (inyectado)
        cache: Cache de resultados (inyectado)
        persistence: Persistencia de análisis (inyectado, opcional)
        plasticity: Puerto de plasticidad (inyectado, opcional)
    """
    
    def __init__(
        self,
        engine: AnalysisEnginePort,
        cache: CachePort,
        persistence: Optional[DocumentPersistencePort] = None,
        plasticity: Optional[PlasticityPort] = None,
    ) -> None:
        """Inicializa caso de uso con dependencias inyectadas."""
        self._engine = engine
        self._cache = cache
        self._persistence = persistence
        self._plasticity = plasticity
    
    def execute(
        self,
        document_id: str,
        content: str,
        tenant_id: str,
        content_type: str = "text",
        filename: Optional[str] = None,
    ) -> AnalysisOutput:
        """Ejecuta análisis con cache y persistencia.
        
        Args:
            document_id: ID único del documento
            content: Contenido a analizar
            tenant_id: ID del tenant (multi-tenancy)
            content_type: Tipo de contenido ('text', 'tabular', etc.)
            filename: Nombre del archivo (opcional)
        
        Returns:
            AnalysisOutput con resultado del análisis
        """
        start = time.time()
        
        # 1. Check cache
        content_hash = compute_content_hash(content)
        cache_key = build_cache_key(content_hash, content_type)
        
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info(f"analysis_cache_hit: document_id={document_id}")
            # Update metadata
            cached.document_id = document_id
            cached.processing_time_ms = round((time.time() - start) * 1000, 2)
            cached.cached = True
            return cached
        
        logger.info(f"analysis_cache_miss: document_id={document_id}")
        
        # 2. Analyze with engine
        try:
            context = AnalysisContext(
                tenant_id=tenant_id,
                series_id=document_id,
                input_type=self._detect_input_type(content_type),
                metadata={"filename": filename} if filename else {},
            )
            
            result = self._engine.analyze(content, context)
            
            # 3. Build output
            output = AnalysisOutput.from_result(result, document_id, tenant_id)
            output.processing_time_ms = round((time.time() - start) * 1000, 2)
            
            # 4. Persist (if port available)
            if self._persistence is not None:
                try:
                    self._persistence.save_analysis(output)
                    logger.debug(f"analysis_persisted: document_id={document_id}")
                except Exception as e:
                    logger.error(f"persistence_failed: {e}", exc_info=True)
            
            # 5. Update plasticity (if port available)
            update_plasticity_from_result(result, self._plasticity)
            
            # 6. Store in cache
            try:
                self._cache.set(cache_key, output)
            except Exception as e:
                logger.warning(f"cache_store_failed: {e}")
            
            return output
        
        except Exception as e:
            logger.error(
                f"analysis_failed: document_id={document_id}, error={e}",
                exc_info=True,
            )
            # Return error output
            return self._build_error_output(document_id, tenant_id, str(e), start)
    
    def _detect_input_type(self, content_type: str) -> InputType:
        """Detecta InputType desde content_type string."""
        type_map = {
            "text": InputType.TEXT,
            "tabular": InputType.TABULAR,
            "document": InputType.DOCUMENT,
            "mixed": InputType.MIXED,
            "timeseries": InputType.TIMESERIES,
        }
        return type_map.get(content_type.lower(), InputType.UNKNOWN)
    
    def _build_error_output(
        self,
        document_id: str,
        tenant_id: str,
        error: str,
        start_time: float,
    ) -> AnalysisOutput:
        """Construye output de error."""
        return AnalysisOutput(
            document_id=document_id,
            tenant_id=tenant_id,
            classification="error",
            conclusion=f"Analysis failed: {error}",
            confidence=0.0,
            analysis={"error": error},
            processing_time_ms=round((time.time() - start_time) * 1000, 2),
        )
