"""Fase 2: Análisis multi-perspectiva — colecta percepciones."""

from __future__ import annotations

import logging
import time
import numpy as np
from typing import Any, Dict, List, Protocol

from iot_machine_learning.domain.ports.analysis import (
    AnalysisContext,
    InputType,
    Perception,
    Signal,
)

logger = logging.getLogger(__name__)


class PerceptionCollector(Protocol):
    """Protocolo para colectores de percepción."""
    
    def collect(self, signal: Signal, context: AnalysisContext) -> List[Perception]:
        """Colecta percepciones del signal."""
        ...


class AnalyzePhase:
    """Fase 2: Análisis multi-perspectiva.
    
    Responsabilidad: colectar percepciones de analizadores específicos por tipo.
    
    Args:
        collectors: Dict de colectores por tipo de input (inyectados)
    """
    
    def __init__(
        self,
        collectors: Dict[InputType, PerceptionCollector],
    ) -> None:
        """Inicializa fase con colectores inyectados."""
        self._collectors = collectors
    
    def execute(
        self,
        signal: Signal,
        context: AnalysisContext,
        timing: Dict[str, float],
    ) -> List[Perception]:
        """Ejecuta análisis multi-perspectiva.
        
        Args:
            signal: Señal percibida
            context: Contexto de análisis
            timing: Dict para registrar tiempos
        
        Returns:
            Lista de percepciones de diferentes analizadores
        """
        t0 = time.monotonic()
        
        # Obtener collector apropiado para el tipo de input
        collector = self._collectors.get(signal.input_type)
        
        if collector is None:
            logger.warning(
                f"no_collector_for_type: {signal.input_type.value}, "
                f"available: {list(self._collectors.keys())}"
            )
            perceptions = []
        else:
            try:
                perceptions = collector.collect(signal, context)
            except Exception as e:
                logger.error(f"perception_collection_failed: {e}", exc_info=True)
                perceptions = []
        
        # Vectorizar operaciones de normalización si hay múltiples percepciones
        if len(perceptions) > 1:
            perceptions = self._normalize_perceptions(perceptions)
        
        timing["analyze"] = (time.monotonic() - t0) * 1000
        
        logger.debug(
            "analyze_complete",
            extra={
                "n_perceptions": len(perceptions),
                "perspectives": [p.perspective for p in perceptions],
                "ms": round(timing["analyze"], 2),
            },
        )
        
        return perceptions
    
    def _normalize_perceptions(
        self,
        perceptions: List[Perception],
    ) -> List[Perception]:
        """Normaliza scores de percepciones usando numpy.
        
        Reemplaza bucles Python por operaciones vectorizadas.
        """
        if not perceptions:
            return perceptions
        
        # Extraer scores como array numpy
        scores = np.array([p.score for p in perceptions])
        confidences = np.array([p.confidence for p in perceptions])
        
        # Normalizar scores (0-1) si hay varianza
        if np.std(scores) > 0:
            scores_normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        else:
            scores_normalized = scores
        
        # Normalizar confidences (0-1)
        if np.std(confidences) > 0:
            confidences_normalized = (confidences - np.min(confidences)) / (np.max(confidences) - np.min(confidences))
        else:
            confidences_normalized = confidences
        
        # Reconstruir percepciones con scores normalizados
        normalized = []
        for i, p in enumerate(perceptions):
            normalized.append(
                Perception(
                    perspective=p.perspective,
                    score=float(scores_normalized[i]),
                    confidence=float(confidences_normalized[i]),
                    evidence=p.evidence,
                    metadata={
                        **p.metadata,
                        "original_score": float(scores[i]),
                        "original_confidence": float(confidences[i]),
                    },
                )
            )
        
        return normalized
