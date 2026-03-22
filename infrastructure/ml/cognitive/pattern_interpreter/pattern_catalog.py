"""Catalog of known patterns with human interpretations."""

from __future__ import annotations

from typing import Dict, Any


PATTERN_CATALOG: Dict[str, Dict[str, Any]] = {
    # Text patterns
    "narrative_escalation": {
        "short_name": "Escalada narrativa",
        "description": "El documento muestra progresión de alertas menores hacia incidente crítico. "
                      "Típico de situaciones que no recibieron atención temprana.",
        "severity_hint": "warning"
    },
    "critical_spike": {
        "short_name": "Pico crítico",
        "description": "Se detectó un aumento abrupto de urgencia en un punto específico del texto. "
                      "Indica momento exacto donde el problema escala.",
        "severity_hint": "critical"
    },
    "sustained_degradation": {
        "short_name": "Degradación sostenida",
        "description": "Urgencia o negatividad consistentemente alta a lo largo del documento. "
                      "Sin puntos de mejora detectados.",
        "severity_hint": "critical"
    },
    "regime_shift": {
        "short_name": "Cambio de contexto",
        "description": "El documento cambia de tema o tono operacional. "
                      "Posible transición entre secciones de diferente criticidad.",
        "severity_hint": "info"
    },
    "stable_operations": {
        "short_name": "Operación estable",
        "description": "No se detectaron cambios significativos de urgencia o régimen. "
                      "Documento informativo sin alertas críticas.",
        "severity_hint": "info"
    },
    
    # Numeric patterns
    "anomalous_spike": {
        "short_name": "Spike anómalo",
        "description": "Valor atípico detectado que supera significativamente el comportamiento normal. "
                      "Requiere verificación inmediata.",
        "severity_hint": "critical"
    },
    "drift_detected": {
        "short_name": "Deriva acumulativa",
        "description": "Desviación gradual pero consistente del valor esperado. "
                      "Si no se corrige puede derivar en fallo.",
        "severity_hint": "warning"
    },
    "operational_regime_change": {
        "short_name": "Cambio de régimen operacional",
        "description": "El sistema cambió de modo de operación (idle→active, normal→peak). "
                      "Verificar si el cambio es esperado o anómalo.",
        "severity_hint": "warning"
    },
    "noise_dominant": {
        "short_name": "Señal ruidosa",
        "description": "Alta variabilidad sin patrón claro. "
                      "Puede indicar inestabilidad o problema de medición.",
        "severity_hint": "warning"
    },
    
    # Change point patterns
    "level_shift": {
        "short_name": "Cambio de nivel",
        "description": "La serie cambió abruptamente a un nuevo nivel de operación. "
                      "Puede indicar reconfiguración del sistema o evento inesperado.",
        "severity_hint": "warning"
    },
    "trend_change": {
        "short_name": "Cambio de tendencia",
        "description": "La dirección de la serie se invirtió o modificó significativamente. "
                      "Requiere análisis de causa raíz.",
        "severity_hint": "warning"
    },
    "variance_change": {
        "short_name": "Cambio de variabilidad",
        "description": "La estabilidad del sistema cambió, volviéndose más o menos predecible. "
                      "Puede afectar la confiabilidad de futuras mediciones.",
        "severity_hint": "info"
    },
    
    # Delta spike patterns
    "delta_spike": {
        "short_name": "Cambio legítimo",
        "description": "Cambio abrupto pero persistente en el nivel de operación. "
                      "Representa una transición real del sistema, no ruido.",
        "severity_hint": "warning"
    },
    "noise_spike": {
        "short_name": "Ruido transitorio",
        "description": "Valor atípico aislado que no persiste. "
                      "Generalmente causado por interferencia o error de medición.",
        "severity_hint": "info"
    },
    
    # General patterns
    "stable": {
        "short_name": "Operación estable",
        "description": "Sin cambios significativos detectados. "
                      "El sistema opera dentro de parámetros normales.",
        "severity_hint": "info"
    },
    "drifting": {
        "short_name": "Deriva gradual",
        "description": "Tendencia gradual away from baseline. "
                      "Puede indicar desgaste o necesidad de recalibración.",
        "severity_hint": "warning"
    },
    "oscillating": {
        "short_name": "Oscilación",
        "description": "Comportamiento cíclico detectado. "
                      "Puede ser normal o indicar inestabilidad según el dominio.",
        "severity_hint": "info"
    },
}


def get_pattern_catalog_entry(pattern_type: str) -> Dict[str, Any]:
    """Get catalog entry for a pattern type.
    
    Args:
        pattern_type: Technical pattern identifier
        
    Returns:
        Catalog entry or default entry if not found
    """
    return PATTERN_CATALOG.get(pattern_type, {
        "short_name": pattern_type.replace("_", " ").title(),
        "description": f"Patrón detectado: {pattern_type}",
        "severity_hint": "info"
    })


def list_all_patterns() -> Dict[str, str]:
    """Get all available patterns with their short names.
    
    Returns:
        Dict mapping pattern_type to short_name
    """
    return {k: v["short_name"] for k, v in PATTERN_CATALOG.items()}
