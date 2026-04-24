"""Pattern creation and description enrichment."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .pattern_catalog import PATTERN_CATALOG
from .types import InterpretedPattern

logger = logging.getLogger(__name__)


def create_pattern(pattern_type: str, domain: str, context_data: Any = None, system_confidence: Optional[float] = None) -> InterpretedPattern:
    """Create interpreted pattern with domain context."""
    logger.info(f"[PATTERN_DEBUG] Looking for pattern: {pattern_type}")
    logger.info(f"[PATTERN_DEBUG] Catalog keys: {list(PATTERN_CATALOG.keys())}")
    logger.info(f"[PATTERN_DEBUG] stable_operations in catalog: {'stable_operations' in PATTERN_CATALOG}")

    if pattern_type in PATTERN_CATALOG:
        catalog = PATTERN_CATALOG[pattern_type]
        logger.info(f"[PATTERN_DEBUG] Found catalog entry: {catalog.get('short_name', 'NO_SHORT_NAME')}")
    else:
        logger.warning(f"[PATTERN_DEBUG] Pattern {pattern_type} not found, using default")
        catalog = {
            "short_name": pattern_type.replace("_", " ").title(),
            "description": f"Patrón detectado: {pattern_type}",
            "severity_hint": "info"
        }

    confidence = system_confidence if system_confidence is not None else 0.8
    description = enrich_description(pattern_type, catalog["description"], context_data)

    return InterpretedPattern(
        pattern_type=pattern_type,
        short_name=catalog["short_name"],
        description=description,
        severity_hint=catalog["severity_hint"],
        domain_context=enrich_domain_context(pattern_type, domain),
        confidence=confidence,
        data_type="text"
    )


def enrich_description(pattern_type: str, base_description: str, context_data: Any) -> str:
    """Enrich base catalog description with real context signals."""
    if not isinstance(context_data, dict):
        return base_description

    enrichments = []

    if pattern_type == "narrative_escalation":
        seg_urgencies = context_data.get("segment_urgencies", [])
        spread = context_data.get("escalation_keywords_spread", 0)
        if seg_urgencies and len(seg_urgencies) >= 2:
            enrichments.append(
                f"Progresión detectada en {len(seg_urgencies)} segmentos "
                f"(urgencia: {min(seg_urgencies):.2f} → {max(seg_urgencies):.2f})."
            )
        if spread >= 2:
            enrichments.append(f"Palabras clave de escalada presentes en {spread} segmentos.")

    elif pattern_type == "critical_spike":
        max_urgency = context_data.get("max_segment_urgency", 0.0)
        min_urgency = context_data.get("min_segment_urgency", 0.0)
        if max_urgency > 0:
            enrichments.append(
                f"Pico crítico detectado: urgencia saltó de {min_urgency:.2f} a {max_urgency:.2f}."
            )

    elif pattern_type == "sustained_degradation":
        high_segments = context_data.get("high_urgency_segments", 0)
        if high_segments > 0:
            enrichments.append(
                f"Degradación sostenida en {high_segments} segmentos consecutivos sin mejoría."
            )

    elif pattern_type == "regime_shift":
        n_change = context_data.get("n_change_points", 0)
        sentiment_shift = context_data.get("sentiment_shift_detected", False)
        if n_change > 0:
            enrichments.append(f"{n_change} cambio(s) estructural(es) detectado(s) en el texto.")
        if sentiment_shift:
            enrichments.append("Cambio de tono detectado: de neutral/positivo a negativo.")

    elif pattern_type == "stable_operations":
        n_segments = len(context_data.get("segment_urgencies", []))
        if n_segments > 0:
            enrichments.append(
                f"Operación estable confirmada en {n_segments} segmentos analizados."
            )

    if enrichments:
        return base_description + " " + " ".join(enrichments)
    return base_description


def enrich_domain_context(pattern_type: str, domain: str) -> str:
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
