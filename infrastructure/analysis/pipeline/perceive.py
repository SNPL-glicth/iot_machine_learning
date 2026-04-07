"""Fase 1: Percepción de señal — detecta tipo y construye perfil."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Protocol

from iot_machine_learning.domain.ports.analysis import (
    AnalysisContext,
    InputType,
    Signal,
)

logger = logging.getLogger(__name__)


class TypeDetector(Protocol):
    """Protocolo para detectores de tipo de input."""
    
    def can_handle(self, data: Any) -> bool:
        """Verifica si puede manejar el dato."""
        ...
    
    def detect_type(self, data: Any) -> InputType:
        """Detecta el tipo de input."""
        ...


class DomainClassifier(Protocol):
    """Protocolo para clasificadores de dominio."""
    
    def classify(self, data: Any, input_type: InputType) -> str:
        """Clasifica el dominio del dato."""
        ...


class FeatureExtractor(Protocol):
    """Protocolo para extractores de features."""
    
    def extract(self, data: Any, input_type: InputType) -> Dict[str, Any]:
        """Extrae features del dato."""
        ...


class PerceivePhase:
    """Fase 1: Percepción de señal.
    
    Responsabilidad: detectar tipo de input, clasificar dominio, construir Signal.
    
    Args:
        type_detectors: Lista de detectores de tipo (inyectados)
        domain_classifier: Clasificador de dominio (inyectado)
        feature_extractor: Extractor de features (inyectado)
    """
    
    def __init__(
        self,
        type_detectors: List[TypeDetector],
        domain_classifier: DomainClassifier,
        feature_extractor: FeatureExtractor,
    ) -> None:
        """Inicializa fase con detectores inyectados."""
        self._type_detectors = type_detectors
        self._domain_classifier = domain_classifier
        self._feature_extractor = feature_extractor
    
    def execute(
        self,
        raw_data: Any,
        context: AnalysisContext,
        timing: Dict[str, float],
    ) -> Signal:
        """Ejecuta percepción de señal.
        
        Args:
            raw_data: Dato de entrada
            context: Contexto de análisis
            timing: Dict para registrar tiempos
        
        Returns:
            Signal con tipo, dominio y features
        """
        t0 = time.monotonic()
        
        # 1. Detectar tipo de input
        input_type = self._detect_type(raw_data, context)
        
        # 2. Clasificar dominio
        domain = self._classify_domain(raw_data, input_type, context)
        
        # 3. Extraer features
        features = self._extract_features(raw_data, input_type)
        
        signal = Signal(
            raw_data=raw_data,
            input_type=input_type,
            domain=domain,
            features=features,
            metadata={
                "tenant_id": context.tenant_id,
                "series_id": context.series_id,
            },
        )
        
        timing["perceive"] = (time.monotonic() - t0) * 1000
        
        logger.debug(
            "perceive_complete",
            extra={
                "input_type": input_type.value,
                "domain": domain,
                "n_features": len(features),
                "ms": round(timing["perceive"], 2),
            },
        )
        
        return signal
    
    def _detect_type(
        self,
        data: Any,
        context: AnalysisContext,
    ) -> InputType:
        """Detecta tipo de input usando detectores inyectados."""
        # Si el contexto ya tiene tipo, usarlo
        if context.input_type is not None:
            return context.input_type
        
        # Probar detectores en orden
        for detector in self._type_detectors:
            if detector.can_handle(data):
                return detector.detect_type(data)
        
        # Fallback: heurística simple
        if isinstance(data, str):
            return InputType.TEXT
        elif isinstance(data, (list, tuple)) and all(isinstance(x, (int, float)) for x in data):
            return InputType.TIMESERIES
        elif isinstance(data, dict):
            return InputType.TABULAR
        else:
            return InputType.UNKNOWN
    
    def _classify_domain(
        self,
        data: Any,
        input_type: InputType,
        context: AnalysisContext,
    ) -> str:
        """Clasifica dominio del dato."""
        # Si el contexto tiene hint, usarlo
        if context.domain_hint:
            return context.domain_hint
        
        # Usar clasificador inyectado
        try:
            return self._domain_classifier.classify(data, input_type)
        except Exception as e:
            logger.warning(f"domain_classification_failed: {e}")
            return "general"
    
    def _extract_features(
        self,
        data: Any,
        input_type: InputType,
    ) -> Dict[str, Any]:
        """Extrae features del dato."""
        try:
            return self._feature_extractor.extract(data, input_type)
        except Exception as e:
            logger.warning(f"feature_extraction_failed: {e}")
            return {}
