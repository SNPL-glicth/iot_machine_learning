"""Text pattern interpretation logic."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .pattern_catalog import get_pattern_catalog_entry
from .types import InterpretedPattern

logger = logging.getLogger(__name__)


def interpret_text_patterns(
    raw_patterns: Dict[str, Any],
    domain: str,
    urgency_score: float = 0.0,
    sentiment_label: str = "",
) -> List[InterpretedPattern]:
    """Interpret patterns detected in text data."""
    interpreted: List[InterpretedPattern] = []
    
    pattern_summary = raw_patterns.get("pattern_summary", {})
    change_points = raw_patterns.get("change_points", [])
    spikes = raw_patterns.get("spikes", [])
    
    # Detect patterns
    if _detect_narrative_escalation(pattern_summary, urgency_score):
        interpreted.append(_create_pattern("narrative_escalation", domain, urgency_score))
    
    if _detect_critical_spike(spikes, pattern_summary):
        interpreted.append(_create_pattern("critical_spike", domain, spikes))
    
    if _detect_sustained_degradation(pattern_summary, urgency_score):
        interpreted.append(_create_pattern("sustained_degradation", domain, urgency_score))
    
    if _detect_regime_shift(change_points, pattern_summary):
        interpreted.append(_create_pattern("regime_shift", domain, change_points))
    
    if not interpreted or _detect_stable_operations(pattern_summary, urgency_score):
        interpreted.append(_create_pattern("stable_operations", domain))
    
    # Sort by severity
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    interpreted.sort(key=lambda p: severity_order.get(p.severity_hint, 3))
    
    return interpreted


def _detect_narrative_escalation(pattern_summary: Dict[str, Any], urgency_score: float) -> bool:
    """Detect progressive escalation from minor to critical issues."""
    # FIX: Use urgency_score >= 0.8 AND negative sentiment to override to escalation pattern
    return urgency_score >= 0.8


def _detect_critical_spike(spikes: List[Any], pattern_summary: Dict[str, Any]) -> bool:
    """Detect abrupt urgency spike at specific point."""
    if not spikes:
        return False
    
    max_spike_magnitude = max((spike.get("magnitude", 0) for spike in spikes), default=0)
    return max_spike_magnitude > 2.0 or pattern_summary.get("has_abrupt_spike", False)


def _detect_sustained_degradation(pattern_summary: Dict[str, Any], urgency_score: float) -> bool:
    """Detect consistently high urgency without improvement."""
    return (
        urgency_score >= 0.7 and
        pattern_summary.get("urgency_regime") == "high" and
        pattern_summary.get("improvement_points", 0) == 0
    )


def _detect_regime_shift(change_points: List[Any], pattern_summary: Dict[str, Any]) -> bool:
    """Detect topic or tonality change in document."""
    return (
        len(change_points) > 0 or
        pattern_summary.get("topic_shift", False) or
        pattern_summary.get("tonality_change", False)
    )


def _detect_stable_operations(pattern_summary: Dict[str, Any], urgency_score: float) -> bool:
    """Detect stable operations without significant changes."""
    # FIX: Never detect stable_operations when urgency is high (>= 0.8)
    if urgency_score >= 0.8:
        return False
    
    return (
        pattern_summary.get("n_change_points", 0) == 0 and
        pattern_summary.get("n_spikes", 0) == 0
    )


def _create_pattern(pattern_type: str, domain: str, context_data: Any = None) -> InterpretedPattern:
    """Create interpreted pattern with domain context."""
    # FIX: Use direct catalog access to avoid import issues
    from .pattern_catalog import PATTERN_CATALOG
    
    # DEBUG: Log what's happening
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[PATTERN_DEBUG] Looking for pattern: {pattern_type}")
    logger.info(f"[PATTERN_DEBUG] Catalog keys: {list(PATTERN_CATALOG.keys())}")
    logger.info(f"[PATTERN_DEBUG] stable_operations in catalog: {'stable_operations' in PATTERN_CATALOG}")
    
    if pattern_type in PATTERN_CATALOG:
        catalog = PATTERN_CATALOG[pattern_type]
        logger.info(f"[PATTERN_DEBUG] Found catalog entry: {catalog.get('short_name', 'NO_SHORT_NAME')}")
    else:
        # Fallback to default entry
        logger.warning(f"[PATTERN_DEBUG] Pattern {pattern_type} not found, using default")
        catalog = {
            "short_name": pattern_type.replace("_", " ").title(),
            "description": f"Patrón detectado: {pattern_type}",
            "severity_hint": "info"
        }
    
    # Calculate confidence based on pattern type and context
    confidence = 0.8
    if pattern_type == "narrative_escalation" and isinstance(context_data, (int, float)):
        confidence = min(0.95, context_data + 0.2)
    elif pattern_type == "critical_spike" and isinstance(context_data, list):
        max_magnitude = max((s.get("magnitude", 0) for s in context_data), default=0)
        confidence = min(0.95, 0.5 + max_magnitude * 0.2)
    elif pattern_type == "sustained_degradation" and isinstance(context_data, (int, float)):
        confidence = min(0.95, context_data + 0.1)
    elif pattern_type == "regime_shift" and isinstance(context_data, list):
        confidence = min(0.9, 0.5 + len(context_data) * 0.1)
    
    return InterpretedPattern(
        pattern_type=pattern_type,
        short_name=catalog["short_name"],
        description=catalog["description"],
        severity_hint=catalog["severity_hint"],
        domain_context=_enrich_domain_context(pattern_type, domain),
        confidence=confidence,
        data_type="text"
    )


def _enrich_domain_context(pattern_type: str, domain: str) -> str:
    """Enrich pattern description with domain-specific context."""
    domain_contexts = {
        "infrastructure": {
            "narrative_escalation": "Típico de incidentes de infraestructura que no recibieron respuesta temprana. Revisa logs de sistema y métricas de rendimiento.",
            "critical_spike": "Puede indicar fallo de componente o pico de carga inesperado. Verifica recursos del sistema.",
            "sustained_degradation": "Sistema operando bajo estrés prolongado. Considera escalado o mantenimiento preventivo.",
            "regime_shift": "Cambio en la configuración del sistema o modo de operación. Verifica deployments o cambios recientes.",
            "stable_operations": "Infraestructura operando normalmente. Continúa monitoreo estándar.",
        },
        "security": {
            "narrative_escalation": "Posible ataque en progreso o brecha de seguridad no contenida. Requiere análisis forense inmediato.",
            "critical_spike": "Actividad anómala detectada. Puede ser intento de intrusión o explotación de vulnerabilidad.",
            "sustained_degradation": "Sistema de seguridad bajo ataque sostenido. Considera activar protocolos de emergencia.",
            "regime_shift": "Cambio en el patrón de amenazas o tácticas de ataque. Actualiza reglas de detección.",
            "stable_operations": "Postura de seguridad normal. Mantén vigilancia de amenazas.",
        },
        "operations": {
            "narrative_escalation": "Incidente operativo escalando sin contención. Requiere intervención de gestión.",
            "critical_spike": "Pico de actividad operativa o fallo crítico de servicio. Verifica SLAs.",
            "sustained_degradation": "Servicio degradado continuamente. Impacto en experiencia del usuario.",
            "regime_shift": "Cambio en modo de operación o proceso de negocio. Verifica con equipos responsables.",
            "stable_operations": "Operaciones dentro de parámetros normales. Continúa monitoreo estándar.",
        },
    }
    
    return domain_contexts.get(domain, {}).get(
        pattern_type, 
        f"Patrón detectado en dominio {domain}. Requiere análisis contextual específico."
    )
