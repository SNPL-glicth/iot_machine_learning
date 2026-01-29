"""Decision context models.

Extracted from decision_context.py for modularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ActionUrgency(Enum):
    """Niveles de urgencia para acciones recomendadas."""
    IMMEDIATE = "immediate"      # Actuar ahora
    PRIORITY = "priority"        # Actuar en las próximas horas
    SCHEDULED = "scheduled"      # Programar para próximo mantenimiento
    MONITOR = "monitor"          # Solo monitorear
    NONE = "none"                # Sin acción requerida


class ImpactLevel(Enum):
    """Niveles de impacto si no se actúa."""
    CRITICAL = "critical"        # Pérdida de servicio, daño a equipos
    HIGH = "high"                # Degradación significativa
    MEDIUM = "medium"            # Degradación menor
    LOW = "low"                  # Impacto mínimo
    NONE = "none"                # Sin impacto esperado


@dataclass
class RecommendedAction:
    """Acción recomendada específica y contextualizada."""
    action: str                          # Descripción de la acción
    urgency: ActionUrgency               # Nivel de urgencia
    assigned_role: str                   # Rol responsable (técnico, operador, etc.)
    estimated_time_minutes: int          # Tiempo estimado para completar
    equipment_needed: list[str] = field(default_factory=list)  # Equipamiento necesario
    preconditions: list[str] = field(default_factory=list)     # Condiciones previas
    
    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "urgency": self.urgency.value,
            "assigned_role": self.assigned_role,
            "estimated_time_minutes": self.estimated_time_minutes,
            "equipment_needed": self.equipment_needed,
            "preconditions": self.preconditions,
        }


@dataclass
class ImpactAssessment:
    """Evaluación de impacto si no se actúa."""
    level: ImpactLevel
    description: str
    time_to_impact_minutes: Optional[int]  # Tiempo estimado hasta el impacto
    affected_systems: list[str] = field(default_factory=list)
    potential_damage: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "description": self.description,
            "time_to_impact_minutes": self.time_to_impact_minutes,
            "affected_systems": self.affected_systems,
            "potential_damage": self.potential_damage,
        }


@dataclass
class EscalationInfo:
    """Información de escalamiento."""
    should_escalate: bool
    escalation_level: int  # 1 = supervisor, 2 = gerente, 3 = dirección
    reason: str
    notify_roles: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "should_escalate": self.should_escalate,
            "escalation_level": self.escalation_level,
            "reason": self.reason,
            "notify_roles": self.notify_roles,
        }


@dataclass
class DecisionContext:
    """Contexto completo para tomar decisiones accionables."""
    
    # Identificación
    sensor_id: int
    device_id: int
    sensor_type: str
    location: str
    
    # Predicción actual
    predicted_value: float
    current_value: float
    trend: str
    confidence: float
    prediction_horizon_minutes: int
    
    # Evaluación
    severity: str
    risk_level: str
    anomaly_detected: bool
    anomaly_score: float
    
    # Decisiones
    recommended_actions: list[RecommendedAction]
    impact_assessment: Optional[ImpactAssessment]
    escalation: Optional[EscalationInfo]
    
    # Resumen
    summary: str
    detailed_analysis: str
    
    # Metadata
    generated_at: datetime
    
    def to_dict(self) -> dict:
        return {
            "sensor": {
                "id": self.sensor_id,
                "type": self.sensor_type,
                "location": self.location,
            },
            "prediction": {
                "current_value": self.current_value,
                "predicted_value": self.predicted_value,
                "trend": self.trend,
                "confidence": self.confidence,
                "horizon_minutes": self.prediction_horizon_minutes,
            },
            "evaluation": {
                "severity": self.severity,
                "risk_level": self.risk_level,
                "anomaly_detected": self.anomaly_detected,
                "anomaly_score": self.anomaly_score,
            },
            "actions": [a.to_dict() for a in self.recommended_actions],
            "impact": self.impact_assessment.to_dict() if self.impact_assessment else None,
            "escalation": self.escalation.to_dict() if self.escalation else None,
            "summary": self.summary,
            "detailed_analysis": self.detailed_analysis,
            "generated_at": self.generated_at.isoformat(),
        }
