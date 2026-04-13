"""Fase 3: Razonamiento — inhibe, adapta, fusiona percepciones."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Protocol

from iot_machine_learning.domain.ports.analysis import (
    AnalysisContext,
    Decision,
    Perception,
    Signal,
)
from .reason_helpers import (
    fuse_perceptions_simple,
    classify_severity_simple,
    build_fallback_decision,
)

logger = logging.getLogger(__name__)


class InhibitionGate(Protocol):
    """Protocolo para inhibidor de percepciones no confiables."""
    
    def filter(self, perceptions: List[Perception]) -> List[Perception]:
        """Filtra percepciones inhibidas."""
        ...


class PlasticityTracker(Protocol):
    """Protocolo para tracker de plasticidad (aprendizaje adaptativo)."""
    
    def get_weights(
        self,
        domain: str,
        perspectives: List[str],
    ) -> Dict[str, float]:
        """Obtiene pesos adaptativos por dominio."""
        ...


class WeightedFusion(Protocol):
    """Protocolo para fusión ponderada de percepciones."""
    
    def fuse(
        self,
        perceptions: List[Perception],
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """Fusiona percepciones con pesos."""
        ...


class SeverityClassifier(Protocol):
    """Protocolo para clasificador de severidad."""
    
    def classify(
        self,
        fused_score: float,
        domain: str,
        perceptions: List[Perception],
    ) -> str:
        """Clasifica severidad del resultado."""
        ...


class ReasonPhase:
    """Fase 3: Razonamiento cognitivo.
    
    Responsabilidad: inhibir, adaptar pesos, fusionar percepciones.
    
    Args:
        inhibitor: Inhibidor de percepciones (inyectado)
        plasticity: Tracker de plasticidad (inyectado)
        fusion: Fusionador ponderado (inyectado)
        severity_classifier: Clasificador de severidad (inyectado)
    """
    
    def __init__(
        self,
        inhibitor: InhibitionGate,
        plasticity: Optional[PlasticityTracker],
        fusion: WeightedFusion,
        severity_classifier: SeverityClassifier,
    ) -> None:
        """Inicializa fase con componentes inyectados."""
        self._inhibitor = inhibitor
        self._plasticity = plasticity
        self._fusion = fusion
        self._severity_classifier = severity_classifier
    
    def execute(
        self,
        perceptions: List[Perception],
        signal: Signal,
        context: AnalysisContext,
        timing: Dict[str, float],
    ) -> Decision:
        """Ejecuta razonamiento cognitivo.
        
        Args:
            perceptions: Percepciones colectadas
            signal: Señal percibida
            context: Contexto de análisis
            timing: Dict para registrar tiempos
        
        Returns:
            Decision con severidad, confianza y pesos
        """
        t0 = time.monotonic()
        
        if not perceptions:
            return build_fallback_decision(signal)
        
        # 1. Inhibir percepciones no confiables
        try:
            active_perceptions = self._inhibitor.filter(perceptions)
        except Exception as e:
            logger.warning(f"inhibition_failed: {e}")
            active_perceptions = perceptions
        
        if not active_perceptions:
            return build_fallback_decision(signal, "all_inhibited")
        
        # 2. Adaptar pesos por dominio (plasticidad)
        perspectives = [p.perspective for p in active_perceptions]
        weights = {p: 1.0 / len(perspectives) for p in perspectives}  # Default
        if self._plasticity is not None:
            try:
                weights = self._plasticity.get_weights(signal.domain, perspectives)
            except Exception as e:
                logger.warning(f"plasticity_failed: {e}")
        
        # 3. Fusionar percepciones con pesos
        try:
            fusion_result = self._fusion.fuse(active_perceptions, weights)
        except Exception as e:
            logger.warning(f"fusion_failed: {e}")
            fusion_result = fuse_perceptions_simple(active_perceptions, weights)
        
        # 4. Clasificar severidad
        # Extraer urgency_score de las features del signal
        urgency_score = 0.0
        if hasattr(signal, 'features') and signal.features:
            urgency_score = signal.features.get('urgency_score', 0.0)
        
        logger.warning(f"[REASON_PHASE] Classifying severity: fused_score={fusion_result['fused_score']:.3f}, urgency_score={urgency_score:.3f}, max={max(fusion_result['fused_score'], urgency_score):.3f}")
        
        try:
            severity = self._severity_classifier.classify(
                fusion_result["fused_score"],
                signal.domain,
                active_perceptions,
            )
            logger.warning(f"[REASON_PHASE] Custom classifier returned: {severity}")
        except Exception as e:
            logger.warning(f"severity_classification_failed: {e}")
            severity = classify_severity_simple(fusion_result["fused_score"], signal.domain, urgency_score)
            logger.warning(f"[REASON_PHASE] Fallback classifier returned: {severity}")
        
        timing["reason"] = (time.monotonic() - t0) * 1000
        
        decision = Decision(
            severity=severity,
            confidence=fusion_result["confidence"],
            perceptions=active_perceptions,
            weights=weights,
            selected_engine=fusion_result.get("selected_engine"),
            selection_reason=fusion_result.get("selection_reason", ""),
            fusion_method=fusion_result.get("method", "weighted_average"),
            metadata={
                "n_inhibited": len(perceptions) - len(active_perceptions),
                "domain": signal.domain,
                "fused_score": fusion_result["fused_score"],
            },
        )
        
        logger.debug("reason_complete", extra={
            "severity": severity, "confidence": round(decision.confidence, 3),
            "n_active": len(active_perceptions), "ms": round(timing["reason"], 2),
        })
        
        return decision
    
