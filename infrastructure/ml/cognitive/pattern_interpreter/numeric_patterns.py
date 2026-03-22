"""Numeric pattern interpretation logic."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .pattern_catalog import get_pattern_catalog_entry
from .types import InterpretedPattern

logger = logging.getLogger(__name__)


def interpret_numeric_patterns(
    raw_patterns: Dict[str, Any],
    domain: str,
    urgency_score: float = 0.0,
    sentiment_label: str = "",
) -> List[InterpretedPattern]:
    """Interpret patterns detected in numeric data."""
    interpreted: List[InterpretedPattern] = []
    
    change_points = raw_patterns.get("change_points", [])
    delta_spikes = raw_patterns.get("delta_spikes", [])
    operational_regime = raw_patterns.get("operational_regime")
    pattern_result = raw_patterns.get("pattern_result")
    
    # Detect patterns
    if _detect_anomalous_spike(delta_spikes, change_points):
        interpreted.append(_create_pattern("anomalous_spike", domain, delta_spikes, change_points))
    
    if _detect_drift_detected(pattern_result, change_points):
        interpreted.append(_create_pattern("drift_detected", domain, pattern_result))
    
    if _detect_operational_regime_change(operational_regime, change_points):
        interpreted.append(_create_pattern("operational_regime_change", domain, operational_regime))
    
    if _detect_noise_dominant(pattern_result, change_points):
        interpreted.append(_create_pattern("noise_dominant", domain, pattern_result))
    
    if _detect_level_shift(change_points):
        interpreted.append(_create_pattern("level_shift", domain, change_points))
    
    if not interpreted or _detect_stable_numeric(pattern_result, change_points):
        interpreted.append(_create_pattern("stable", domain))
    
    # Sort by severity
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    interpreted.sort(key=lambda p: severity_order.get(p.severity_hint, 3))
    
    return interpreted


def _detect_anomalous_spike(delta_spikes: List[Any], change_points: List[Any]) -> bool:
    """Detect anomalous spikes in numeric data."""
    # Check delta spikes
    if delta_spikes:
        for spike in delta_spikes:
            if spike.get("is_delta_spike", False) and spike.get("delta_magnitude", 0) > 2.0:
                return True
    
    # Check change points
    if change_points:
        return any(cp.get("magnitude", 0) > 3.0 for cp in change_points)
    
    return False


def _detect_drift_detected(pattern_result: Any, change_points: List[Any]) -> bool:
    """Detect gradual drift from expected values."""
    if pattern_result:
        pattern_type = getattr(pattern_result, "pattern_type", None)
        return pattern_type and pattern_type.value in ["drifting", "curve_anomaly"]
    return False


def _detect_operational_regime_change(operational_regime: Any, change_points: List[Any]) -> bool:
    """Detect change in operational regime."""
    return operational_regime is not None and len(change_points) > 0


def _detect_noise_dominant(pattern_result: Any, change_points: List[Any]) -> bool:
    """Detect high variability without clear pattern."""
    if pattern_result:
        pattern_type = getattr(pattern_result, "pattern_type", None)
        return pattern_type and pattern_type.value in ["oscillating", "micro_variation"]
    return False


def _detect_level_shift(change_points: List[Any]) -> bool:
    """Detect abrupt level shift."""
    if not change_points:
        return False
    
    return any(
        getattr(cp, "change_type", None) and getattr(cp, "change_type").value == "level_shift"
        for cp in change_points
    )


def _detect_stable_numeric(pattern_result: Any, change_points: List[Any]) -> bool:
    """Detect stable numeric operations."""
    if pattern_result:
        pattern_type = getattr(pattern_result, "pattern_type", None)
        return pattern_type and pattern_type.value == "stable"
    return len(change_points) == 0


def _create_pattern(pattern_type: str, domain: str, *context_data: Any) -> InterpretedPattern:
    """Create interpreted pattern with domain context."""
    catalog = get_pattern_catalog_entry(pattern_type)
    
    # Calculate confidence based on pattern type and context
    confidence = 0.8
    if pattern_type == "anomalous_spike" and context_data:
        max_magnitude = 0.0
        if context_data[0]:  # delta_spikes
            max_magnitude = max(s.get("delta_magnitude", 0) for s in context_data[0])
        if context_data[1]:  # change_points
            max_magnitude = max(max_magnitude, max(cp.get("magnitude", 0) for cp in context_data[1]))
        confidence = min(0.95, 0.5 + max_magnitude * 0.1)
    elif pattern_type in ["drift_detected", "noise_dominant"] and context_data:
        confidence = getattr(context_data[0], "confidence", 0.7) if context_data[0] else 0.7
    elif pattern_type == "level_shift" and context_data:
        level_shifts = [cp for cp in context_data[0] 
                       if getattr(cp, "change_type", None) and 
                       getattr(cp, "change_type").value == "level_shift"]
        if level_shifts:
            max_magnitude = max(cp.get("magnitude", 0) for cp in level_shifts)
            confidence = min(0.9, 0.5 + max_magnitude * 0.1)
    elif pattern_type == "operational_regime_change" and context_data:
        regime_name = getattr(context_data[0], "name", "unknown") if context_data[0] else "unknown"
        confidence = 0.8
    
    domain_context = _enrich_domain_context(pattern_type, domain, *context_data)
    
    return InterpretedPattern(
        pattern_type=pattern_type,
        short_name=catalog["short_name"],
        description=catalog["description"],
        severity_hint=catalog["severity_hint"],
        domain_context=domain_context,
        confidence=confidence,
        data_type="numeric"
    )


def _enrich_domain_context(pattern_type: str, domain: str, *context_data: Any) -> str:
    """Enrich pattern description with domain-specific context."""
    base_contexts = {
        "infrastructure": {
            "anomalous_spike": "Pico inesperado en métricas de infraestructura. Revisa CPU, memoria, red o almacenamiento.",
            "drift_detected": "Deriva gradual en métricas del sistema. Puede indicar degradación de rendimiento o necesidad de mantenimiento.",
            "operational_regime_change": "Cambio en modo de operación del sistema. Verifica si es esperado o indica problema.",
            "noise_dominant": "Métricas inestables o ruidosas. Puede indicar problema de monitoreo o sistema inestable.",
            "level_shift": "Cambio abrupto en nivel de operación. Posible reconfiguración o evento de sistema.",
            "stable": "Métricas de infraestructura estables. Sistema operando normalmente.",
        },
        "security": {
            "anomalous_spike": "Pico en métricas de seguridad. Puede indicar ataque o brecha. Revisa logs de seguridad.",
            "drift_detected": "Tendencia anómala en indicadores de seguridad. Monitorea posibles compromisos.",
            "operational_regime_change": "Cambio en patrón de actividad de seguridad. Actualiza reglas de detección.",
            "noise_dominant": "Alta variabilidad en métricas de seguridad. Puede indicar evasión o actividad anómala.",
            "level_shift": "Cambio abrupto en nivel de actividad. Verifica eventos de seguridad recientes.",
            "stable": "Métricas de seguridad normales. Continúa monitoreo estándar.",
        },
        "operations": {
            "anomalous_spike": "Pico en métricas operativas. Impacto posible en SLAs o experiencia del usuario.",
            "drift_detected": "Tendencia degradativa en operaciones. Requiere optimización o intervención.",
            "operational_regime_change": "Cambio en modo de operación. Verifica con equipos de operaciones.",
            "noise_dominant": "Métricas operativas inestables. Problema de proceso o sistema.",
            "level_shift": "Cambio en nivel de operación. Posible reconfiguración de servicio.",
            "stable": "Operaciones estables. Métricas dentro de parámetros normales.",
        },
    }
    
    context = base_contexts.get(domain, {}).get(
        pattern_type, 
        f"Patrón numérico detectado en dominio {domain}. Requiere análisis contextual específico."
    )
    
    # Add regime-specific context for operational_regime_change
    if pattern_type == "operational_regime_change" and context_data:
        regime_name = getattr(context_data[0], "name", "unknown") if context_data[0] else "unknown"
        context += f" Régimen actual: {regime_name}."
    
    return context
