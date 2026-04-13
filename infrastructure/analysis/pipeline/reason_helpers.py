"""Helpers para fase de razonamiento — fusión y clasificación."""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List

from iot_machine_learning.domain.ports.analysis import Decision, Perception, Signal


def fuse_perceptions_simple(
    perceptions: List[Perception],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """Fusiona percepciones con pesos usando numpy (fallback simple).
    
    Args:
        perceptions: Lista de percepciones
        weights: Dict de pesos por perspectiva
    
    Returns:
        Dict con fused_score, confidence, method
    """
    if not perceptions:
        return {
            "fused_score": 0.0,
            "confidence": 0.0,
            "method": "empty",
        }
    
    # Vectorizar operaciones
    scores = np.array([p.score for p in perceptions])
    confidences = np.array([p.confidence for p in perceptions])
    weight_values = np.array([
        weights.get(p.perspective, 1.0 / len(perceptions))
        for p in perceptions
    ])
    
    # Normalizar pesos
    if np.sum(weight_values) > 0:
        weight_values = weight_values / np.sum(weight_values)
    
    # Fusión ponderada
    fused_score = float(np.sum(scores * weight_values))
    fused_confidence = float(np.mean(confidences))
    
    # Seleccionar motor con mayor peso
    max_idx = int(np.argmax(weight_values))
    selected = perceptions[max_idx].perspective
    
    return {
        "fused_score": fused_score,
        "confidence": fused_confidence,
        "selected_engine": selected,
        "selection_reason": f"highest_weight: {weight_values[max_idx]:.3f}",
        "method": "simple_average_fallback",
    }


def classify_severity_simple(
    fused_score: float,
    domain: str,
    urgency_score: float = 0.0,
) -> str:
    """Clasifica severidad considerando fused_score y urgency del texto.
    
    Args:
        fused_score: Score fusionado [0-1]
        domain: Dominio del análisis
        urgency_score: Urgencia detectada en el texto [0-1]
        
    Returns:
        Severidad ('info', 'warning', 'critical')
    """
    # Usar el máximo entre fused_score y urgency para no ignorar señales críticas
    max_score = max(fused_score, urgency_score)
    
    # Clasificación con umbrales
    if max_score >= 0.8:
        return "critical"
    elif max_score >= 0.5:
        return "warning"
    else:
        return "info"


def build_fallback_decision(
    signal: Signal,
    reason: str = "no_perceptions",
) -> Decision:
    """Construye decisión fallback cuando no hay percepciones.
    
    Args:
        signal: Señal percibida
        reason: Razón del fallback
    
    Returns:
        Decision con valores por defecto
    """
    return Decision(
        severity="info",
        confidence=0.0,
        perceptions=[],
        weights={},
        selection_reason=f"fallback: {reason}",
        fusion_method="none",
        metadata={"domain": signal.domain, "fallback": True},
    )
