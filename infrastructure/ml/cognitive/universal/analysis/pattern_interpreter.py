"""Pattern Interpreter - Human-readable interpretation of detected patterns.

Transforms technical pattern detection results into human-understandable
interpretations with domain context and severity classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InterpretedPattern:
    """Human-interpretable pattern result.
    
    Transforms technical pattern detection into actionable insights.
    
    Attributes:
        pattern_type: Technical pattern identifier
        short_name: Human-friendly pattern name
        description: Context-aware explanation
        severity_hint: Severity classification
        domain_context: Domain-specific interpretation
        confidence: Confidence in interpretation (0-1)
        data_type: Type of data this pattern applies to
    """
    pattern_type: str          # technical: "cusum_drift", "delta_spike", "regime_change"
    short_name: str            # human: "Escalada crítica", "Spike anómalo", "Cambio de régimen"
    description: str           # full context-aware description
    severity_hint: str         # "info" | "warning" | "critical"
    domain_context: str        # how this pattern relates to the domain
    confidence: float          # [0, 1]
    data_type: str             # "text" | "numeric" | "universal"


# Pattern catalog
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
    """Get catalog entry for a pattern type."""
    return PATTERN_CATALOG.get(pattern_type, {
        "short_name": pattern_type.replace("_", " ").title(),
        "description": f"Patrón detectado: {pattern_type}",
        "severity_hint": "info"
    })


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
        interpreted.append(_create_pattern("narrative_escalation", domain, urgency_score, "text"))
    
    if _detect_critical_spike(spikes, pattern_summary):
        interpreted.append(_create_pattern("critical_spike", domain, spikes, "text"))
    
    if _detect_sustained_degradation(pattern_summary, urgency_score):
        interpreted.append(_create_pattern("sustained_degradation", domain, urgency_score, "text"))
    
    if _detect_regime_shift(change_points, pattern_summary):
        interpreted.append(_create_pattern("regime_shift", domain, change_points, "text"))
    
    if not interpreted or _detect_stable_operations(pattern_summary, urgency_score):
        interpreted.append(_create_pattern("stable_operations", domain, None, "text"))
    
    # Sort by severity
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    interpreted.sort(key=lambda p: severity_order.get(p.severity_hint, 3))
    
    return interpreted


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
        interpreted.append(_create_pattern("anomalous_spike", domain, delta_spikes, "numeric"))
    
    if _detect_drift_detected(pattern_result, change_points):
        interpreted.append(_create_pattern("drift_detected", domain, pattern_result, "numeric"))
    
    if _detect_operational_regime_change(operational_regime, change_points):
        interpreted.append(_create_pattern("operational_regime_change", domain, operational_regime, "numeric"))
    
    if _detect_noise_dominant(pattern_result, change_points):
        interpreted.append(_create_pattern("noise_dominant", domain, pattern_result, "numeric"))
    
    if _detect_level_shift(change_points):
        interpreted.append(_create_pattern("level_shift", domain, change_points, "numeric"))
    
    if not interpreted or _detect_stable_numeric(pattern_result, change_points):
        interpreted.append(_create_pattern("stable", domain, None, "numeric"))
    
    # Sort by severity
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    interpreted.sort(key=lambda p: severity_order.get(p.severity_hint, 3))
    
    return interpreted


def _detect_narrative_escalation(pattern_summary: Dict[str, Any], urgency_score: float) -> bool:
    """Detect progressive escalation from minor to critical issues."""
    has_high_urgency = urgency_score >= 0.8
    has_escalation_indicator = pattern_summary.get("has_escalation", False)
    has_multiple_spikes = pattern_summary.get("n_spikes", 0) > 1
    return has_high_urgency and (has_escalation_indicator or has_multiple_spikes)


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
    if urgency_score >= 0.8:
        return False
    return (
        pattern_summary.get("n_change_points", 0) == 0 and
        pattern_summary.get("n_spikes", 0) == 0
    )


def _detect_anomalous_spike(delta_spikes: List[Any], change_points: List[Any]) -> bool:
    """Detect anomalous spikes in numeric data."""
    if delta_spikes:
        for spike in delta_spikes:
            if spike.get("is_delta_spike", False) and spike.get("delta_magnitude", 0) > 2.0:
                return True
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


def _create_pattern(pattern_type: str, domain: str, context_data: Any, data_type: str) -> InterpretedPattern:
    """Create interpreted pattern with domain context."""
    catalog = get_pattern_catalog_entry(pattern_type)
    
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
    
    domain_context = _enrich_domain_context(pattern_type, domain, data_type)
    
    return InterpretedPattern(
        pattern_type=pattern_type,
        short_name=catalog["short_name"],
        description=catalog["description"],
        severity_hint=catalog["severity_hint"],
        domain_context=domain_context,
        confidence=confidence,
        data_type=data_type
    )


def _enrich_domain_context(pattern_type: str, domain: str, data_type: str) -> str:
    """Enrich pattern description with domain-specific context."""
    domain_contexts = {
        "infrastructure": {
            "narrative_escalation": "Típico de incidentes de infraestructura que no recibieron respuesta temprana. Revisa logs de sistema y métricas de rendimiento.",
            "critical_spike": "Puede indicar fallo de componente o pico de carga inesperado. Verifica recursos del sistema.",
            "sustained_degradation": "Sistema operando bajo estrés prolongado. Considera escalado o mantenimiento preventivo.",
            "regime_shift": "Cambio en la configuración del sistema o modo de operación. Verifica deployments o cambios recientes.",
            "stable_operations": "Infraestructura operando normalmente. Continúa monitoreo estándar.",
            "anomalous_spike": "Pico inesperado en métricas de infraestructura. Revisa CPU, memoria, red o almacenamiento.",
            "drift_detected": "Deriva gradual en métricas del sistema. Puede indicar degradación de rendimiento o necesidad de mantenimiento.",
            "operational_regime_change": "Cambio en modo de operación del sistema. Verifica si es esperado o indica problema.",
            "noise_dominant": "Métricas inestables o ruidosas. Puede indicar problema de monitoreo o sistema inestable.",
            "level_shift": "Cambio abrupto en nivel de operación. Posible reconfiguración o evento de sistema.",
            "stable": "Métricas de infraestructura estables. Sistema operando normalmente.",
        },
        "security": {
            "narrative_escalation": "Posible ataque en progreso o brecha de seguridad no contenida. Requiere análisis forense inmediato.",
            "critical_spike": "Actividad anómala detectada. Puede ser intento de intrusión o explotación de vulnerabilidad.",
            "sustained_degradation": "Sistema de seguridad bajo ataque sostenido. Considera activar protocolos de emergencia.",
            "regime_shift": "Cambio en el patrón de amenazas o tácticas de ataque. Actualiza reglas de detección.",
            "stable_operations": "Postura de seguridad normal. Mantén vigilancia de amenazas.",
            "anomalous_spike": "Pico en métricas de seguridad. Puede indicar ataque o brecha. Revisa logs de seguridad.",
            "drift_detected": "Tendencia anómala en indicadores de seguridad. Monitorea posibles compromisos.",
            "operational_regime_change": "Cambio en patrón de actividad de seguridad. Actualiza reglas de detección.",
            "noise_dominant": "Alta variabilidad en métricas de seguridad. Puede indicar evasión o actividad anómala.",
            "level_shift": "Cambio abrupto en nivel de actividad. Verifica eventos de seguridad recientes.",
            "stable": "Métricas de seguridad normales. Continúa monitoreo estándar.",
        },
        "operations": {
            "narrative_escalation": "Incidente operativo escalando sin contención. Requiere intervención de gestión.",
            "critical_spike": "Pico de actividad operativa o fallo crítico de servicio. Verifica SLAs.",
            "sustained_degradation": "Servicio degradado continuamente. Impacto en experiencia del usuario.",
            "regime_shift": "Cambio en modo de operación o proceso de negocio. Verifica con equipos responsables.",
            "stable_operations": "Operaciones dentro de parámetros normales. Continúa monitoreo estándar.",
            "anomalous_spike": "Pico en métricas operativas. Impacto posible en SLAs o experiencia del usuario.",
            "drift_detected": "Tendencia degradativa en operaciones. Requiere optimización o intervención.",
            "operational_regime_change": "Cambio en modo de operación. Verifica con equipos de operaciones.",
            "noise_dominant": "Métricas operativas inestables. Problema de proceso o sistema.",
            "level_shift": "Cambio en nivel de operación. Posible reconfiguración de servicio.",
            "stable": "Operaciones estables. Métricas dentro de parámetros normales.",
        },
    }
    
    return domain_contexts.get(domain, {}).get(
        pattern_type, 
        f"Patrón detectado en dominio {domain}. Requiere análisis contextual específico."
    )


class PatternInterpreter:
    """Human-readable interpreter of detected patterns."""
    
    def __init__(self) -> None:
        self._text_patterns = interpret_text_patterns
        self._numeric_patterns = interpret_numeric_patterns
    
    def interpret(
        self,
        raw_patterns: Dict[str, Any],
        input_type: str,
        domain: str,
        urgency_score: float = 0.0,
        sentiment_label: str = "",
    ) -> List[InterpretedPattern]:
        """Interpret detected patterns into human-readable format."""
        if raw_patterns is None:
            return []
        
        try:
            if input_type == "text":
                return self._text_patterns(raw_patterns, domain, urgency_score, sentiment_label)
            elif input_type == "numeric":
                return self._numeric_patterns(raw_patterns, domain, urgency_score, sentiment_label)
            elif input_type == "universal":
                return self._merge_patterns(raw_patterns, domain, urgency_score, sentiment_label)
            else:
                logger.warning(f"unknown_input_type", extra={"input_type": input_type})
                return []
        except Exception as e:
            logger.error("pattern_interpretation_failed", extra={"input_type": input_type, "domain": domain, "error": str(e)}, exc_info=True)
            return []
    
    def _merge_patterns(self, raw_patterns: Dict[str, Any], domain: str, urgency_score: float, sentiment_label: str) -> List[InterpretedPattern]:
        """Merge text and numeric pattern results for universal input."""
        text_results = self._text_patterns(raw_patterns, domain, urgency_score, sentiment_label)
        numeric_results = self._numeric_patterns(raw_patterns, domain, urgency_score, sentiment_label)
        
        # Deduplicate by pattern_type
        seen_types = set()
        merged = []
        for pattern in text_results + numeric_results:
            if pattern.pattern_type not in seen_types:
                merged.append(pattern)
                seen_types.add(pattern.pattern_type)
        
        # Sort by severity
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        merged.sort(key=lambda p: (severity_order.get(p.severity_hint, 3), -p.confidence))
        return merged
    
    def get_primary_pattern(self, patterns: List[InterpretedPattern]) -> Optional[InterpretedPattern]:
        """Get the most severe pattern from the list."""
        if not patterns:
            return None
        
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        return min(patterns, key=lambda p: (severity_order.get(p.severity_hint, 3), -p.confidence))
    
    def format_for_conclusion(self, patterns: List[InterpretedPattern], domain: str) -> str:
        """Format patterns as human-readable conclusion string."""
        if not patterns:
            return f"No se detectaron patrones significativos en dominio {domain}."
        
        primary = self.get_primary_pattern(patterns)
        if not primary:
            return "Análisis de patrones no disponible."
        
        conclusion_parts = [f"{primary.short_name}: {primary.description}"]
        
        if primary.domain_context:
            conclusion_parts.append(f"Contexto: {primary.domain_context}")
        
        # Add other critical patterns (max 2)
        critical_patterns = [p for p in patterns if p.severity_hint == "critical" and p.pattern_type != primary.pattern_type][:2]
        if critical_patterns:
            conclusion_parts.append("Otros patrones críticos:")
            conclusion_parts.extend(f"- {p.short_name}" for p in critical_patterns)
        
        conclusion_parts.append(f"Confianza: {int(primary.confidence * 100)}%")
        return "\n".join(conclusion_parts)
    
    def get_pattern_summary(self, patterns: List[InterpretedPattern]) -> Dict[str, Any]:
        """Get summary statistics of interpreted patterns."""
        if not patterns:
            return {"total_patterns": 0, "severity_breakdown": {"critical": 0, "warning": 0, "info": 0}, "data_types": [], "primary_pattern": None}
        
        severity_counts = {"critical": 0, "warning": 0, "info": 0}
        data_types = set()
        
        for pattern in patterns:
            severity_counts[pattern.severity_hint] += 1
            data_types.add(pattern.data_type)
        
        return {
            "total_patterns": len(patterns),
            "severity_breakdown": severity_counts,
            "data_types": list(data_types),
            "primary_pattern": self.get_primary_pattern(patterns),
        }
