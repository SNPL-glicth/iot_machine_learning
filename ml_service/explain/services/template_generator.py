"""Template-based explanation generator.

Extracts template generation logic from ContextualExplainer.
"""

from __future__ import annotations

from datetime import datetime, timezone

from iot_machine_learning.domain.entities.results.anomaly import AnomalySeverity
from ..models.enriched_context import EnrichedContext
from ..models.explanation_result import ExplanationResult


class TemplateExplanationGenerator:
    """Generates explanations using templates when LLM is unavailable."""
    
    # Causas posibles por tipo de sensor
    CAUSES_BY_SENSOR_TYPE = {
        "temperature": {
            "rising": [
                "Falla en sistema de climatización",
                "Aumento de carga térmica (equipos adicionales)",
                "Obstrucción en ventilación",
                "Falla del sensor (descalibración)",
            ],
            "falling": [
                "Climatización en modo máximo",
                "Pérdida de aislamiento térmico",
                "Falla del sensor",
            ],
        },
        "humidity": {
            "rising": [
                "Filtración de agua",
                "Falla en deshumidificador",
                "Condensación por diferencia térmica",
            ],
            "falling": [
                "Ambiente muy seco",
                "Sistema de climatización extrayendo humedad",
            ],
        },
        "power": {
            "default": [
                "Fluctuación en suministro eléctrico",
                "Sobrecarga en circuito",
                "Conexión defectuosa",
                "Falla en equipo conectado",
            ],
        },
        "voltage": {
            "default": [
                "Fluctuación en suministro eléctrico",
                "Sobrecarga en circuito",
                "Conexión defectuosa",
                "Falla en equipo conectado",
            ],
        },
        "air_quality": {
            "default": [
                "Ventilación insuficiente",
                "Acumulación de CO2 por ocupación",
                "Falla en sistema de renovación de aire",
            ],
        },
    }
    
    DEFAULT_CAUSES = [
        "Cambio en condiciones operativas",
        "Posible falla del sensor",
        "Variación ambiental externa",
    ]
    
    def generate(self, context: EnrichedContext) -> ExplanationResult:
        """Genera explicación basada en templates."""
        severity = self._determine_severity(context)
        explanation = self._build_explanation_text(context, severity)
        causes = self._generate_possible_causes(context)
        action = self._generate_recommended_action(context, severity)
        confidence = self._calculate_confidence(context)
        
        return ExplanationResult(
            severity=severity,
            explanation=explanation,
            possible_causes=causes,
            recommended_action=action,
            confidence=confidence,
            source="template",
            generated_at=datetime.now(timezone.utc),
        )
    
    def _determine_severity(self, context: EnrichedContext) -> str:
        """Determina la severidad basada en el contexto.

        Delegates anomaly-score classification to the domain's
        ``AnomalySeverity.from_score()`` — single source of truth
        for severity thresholds (COG-4).
        """
        out_of_range = False
        if context.user_threshold_min is not None:
            out_of_range = context.predicted_value < context.user_threshold_min
        if context.user_threshold_max is not None:
            out_of_range = out_of_range or context.predicted_value > context.user_threshold_max

        if out_of_range:
            return "CRITICAL"

        domain_severity = AnomalySeverity.from_score(context.anomaly_score)

        if context.is_anomaly and domain_severity in (
            AnomalySeverity.HIGH, AnomalySeverity.CRITICAL,
        ):
            return "HIGH"

        if context.is_anomaly or domain_severity in (
            AnomalySeverity.MEDIUM, AnomalySeverity.HIGH, AnomalySeverity.CRITICAL,
        ):
            return "MEDIUM"

        return "LOW"
    
    def _build_explanation_text(self, context: EnrichedContext, severity: str) -> str:
        """Construye el texto de explicación."""
        trend_desc = {
            "rising": "tendencia ascendente",
            "falling": "tendencia descendente",
            "stable": "comportamiento estable",
        }.get(context.trend, context.trend)
        
        delta = context.predicted_value - context.current_value
        delta_pct = (delta / context.current_value * 100) if context.current_value != 0 else 0
        
        parts = [
            f"El sensor {context.sensor_name} ({context.sensor_type}) en {context.location} "
            f"muestra {trend_desc}.",
            f"Valor actual: {context.current_value:.2f} {context.unit}. "
            f"Predicción a {context.horizon_minutes} min: {context.predicted_value:.2f} {context.unit} "
            f"(cambio de {delta:+.2f}, {delta_pct:+.1f}%).",
        ]
        
        if context.is_anomaly:
            parts.append(
                f"Se detectó comportamiento anómalo (score: {context.anomaly_score:.2f})."
            )
        
        if context.similar_events_count > 0:
            parts.append(
                f"Este patrón se ha observado {context.similar_events_count} veces en los últimos 30 días."
            )
        
        if context.correlated_events:
            parts.append(
                f"Hay {len(context.correlated_events)} eventos relacionados en otros sensores del mismo dispositivo."
            )
        
        return " ".join(parts)
    
    def _generate_possible_causes(self, context: EnrichedContext) -> list[str]:
        """Genera lista de causas posibles basadas en el tipo de sensor."""
        sensor_type = context.sensor_type.lower()
        trend = context.trend.lower()
        
        if sensor_type in self.CAUSES_BY_SENSOR_TYPE:
            sensor_causes = self.CAUSES_BY_SENSOR_TYPE[sensor_type]
            if trend in sensor_causes:
                return sensor_causes[trend][:4]
            if "default" in sensor_causes:
                return sensor_causes["default"][:4]
        
        return self.DEFAULT_CAUSES[:4]
    
    def _generate_recommended_action(self, context: EnrichedContext, severity: str) -> str:
        """Genera acción recomendada basada en severidad y contexto."""
        if severity == "CRITICAL":
            return (
                f"ACCIÓN INMEDIATA: Verificar físicamente el sensor {context.sensor_name} "
                f"en {context.location}. Revisar sistemas relacionados y notificar a supervisión."
            )
        
        if severity == "HIGH":
            return (
                f"Programar inspección prioritaria del sensor {context.sensor_name} "
                f"y sistemas asociados en {context.location} dentro de las próximas 2 horas."
            )
        
        if severity == "MEDIUM":
            return (
                f"Monitorear de cerca el sensor {context.sensor_name}. "
                f"Si la tendencia continúa, programar revisión para el próximo turno."
            )
        
        return "Continuar monitoreo normal. No se requiere acción inmediata."
    
    def _calculate_confidence(self, context: EnrichedContext) -> float:
        """Calcula la confianza de la explicación basada en template."""
        base_confidence = 0.6
        
        if context.similar_events_count > 3:
            base_confidence += 0.1
        
        if context.user_threshold_min is not None or context.user_threshold_max is not None:
            base_confidence += 0.1
        
        if context.recent_avg is not None:
            base_confidence += 0.05
        
        return min(0.85, base_confidence)
