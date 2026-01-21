"""Módulo de Contexto de Decisión para ML.

FALENCIA 1: Las predicciones actuales no generan recomendaciones accionables contextualizadas.

Este módulo proporciona:
- Contexto enriquecido para decisiones (historial, ubicación, impacto)
- Acciones específicas y priorizadas
- Estimación de impacto si no se actúa
- Información de escalamiento
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection


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
    """Contexto completo para toma de decisiones.
    
    Este dataclass encapsula toda la información necesaria para que
    un operador o sistema automatizado tome una decisión informada.
    """
    
    # Identificación
    sensor_id: int
    device_id: int
    sensor_type: str
    location: str
    
    # Predicción base
    predicted_value: float
    current_value: float
    trend: str
    confidence: float
    
    # Contexto temporal
    prediction_horizon_minutes: int
    generated_at: datetime
    
    # Análisis de severidad
    severity: str
    risk_level: str
    anomaly_detected: bool
    anomaly_score: float
    
    # Contexto histórico
    similar_events_count: int = 0
    last_similar_event_at: Optional[datetime] = None
    avg_resolution_time_minutes: Optional[int] = None
    common_root_cause: Optional[str] = None
    
    # Acciones recomendadas (priorizadas)
    recommended_actions: list[RecommendedAction] = field(default_factory=list)
    
    # Evaluación de impacto
    impact_assessment: Optional[ImpactAssessment] = None
    
    # Escalamiento
    escalation: Optional[EscalationInfo] = None
    
    # Resumen ejecutivo
    summary: str = ""
    detailed_analysis: str = ""
    
    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "device_id": self.device_id,
            "sensor_type": self.sensor_type,
            "location": self.location,
            "predicted_value": self.predicted_value,
            "current_value": self.current_value,
            "trend": self.trend,
            "confidence": self.confidence,
            "prediction_horizon_minutes": self.prediction_horizon_minutes,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "severity": self.severity,
            "risk_level": self.risk_level,
            "anomaly_detected": self.anomaly_detected,
            "anomaly_score": self.anomaly_score,
            "similar_events_count": self.similar_events_count,
            "last_similar_event_at": self.last_similar_event_at.isoformat() if self.last_similar_event_at else None,
            "avg_resolution_time_minutes": self.avg_resolution_time_minutes,
            "common_root_cause": self.common_root_cause,
            "recommended_actions": [a.to_dict() for a in self.recommended_actions],
            "impact_assessment": self.impact_assessment.to_dict() if self.impact_assessment else None,
            "escalation": self.escalation.to_dict() if self.escalation else None,
            "summary": self.summary,
            "detailed_analysis": self.detailed_analysis,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class DecisionContextBuilder:
    """Builder para construir DecisionContext con información enriquecida."""
    
    def __init__(self, conn: Connection):
        self._conn = conn
    
    def build(
        self,
        *,
        sensor_id: int,
        device_id: int,
        sensor_type: str,
        location: str,
        predicted_value: float,
        current_value: float,
        trend: str,
        confidence: float,
        prediction_horizon_minutes: int,
        severity: str,
        risk_level: str,
        anomaly_detected: bool,
        anomaly_score: float,
    ) -> DecisionContext:
        """Construye un DecisionContext completo con información enriquecida."""
        
        # Obtener historial de eventos similares
        similar_events = self._get_similar_events_history(sensor_id, severity)
        
        # Generar acciones recomendadas contextualizadas
        actions = self._generate_recommended_actions(
            sensor_type=sensor_type,
            location=location,
            severity=severity,
            trend=trend,
            anomaly_detected=anomaly_detected,
        )
        
        # Evaluar impacto
        impact = self._assess_impact(
            sensor_type=sensor_type,
            severity=severity,
            trend=trend,
            predicted_value=predicted_value,
            current_value=current_value,
        )
        
        # Determinar escalamiento
        escalation = self._determine_escalation(
            severity=severity,
            similar_events_count=similar_events.get("count", 0),
            anomaly_detected=anomaly_detected,
        )
        
        # Generar resumen ejecutivo
        summary = self._generate_summary(
            sensor_type=sensor_type,
            location=location,
            severity=severity,
            trend=trend,
            predicted_value=predicted_value,
            similar_events_count=similar_events.get("count", 0),
        )
        
        # Generar análisis detallado
        detailed = self._generate_detailed_analysis(
            sensor_type=sensor_type,
            location=location,
            predicted_value=predicted_value,
            current_value=current_value,
            trend=trend,
            confidence=confidence,
            anomaly_score=anomaly_score,
            similar_events=similar_events,
        )
        
        return DecisionContext(
            sensor_id=sensor_id,
            device_id=device_id,
            sensor_type=sensor_type,
            location=location,
            predicted_value=predicted_value,
            current_value=current_value,
            trend=trend,
            confidence=confidence,
            prediction_horizon_minutes=prediction_horizon_minutes,
            generated_at=datetime.now(timezone.utc),
            severity=severity,
            risk_level=risk_level,
            anomaly_detected=anomaly_detected,
            anomaly_score=anomaly_score,
            similar_events_count=similar_events.get("count", 0),
            last_similar_event_at=similar_events.get("last_at"),
            avg_resolution_time_minutes=similar_events.get("avg_resolution_minutes"),
            common_root_cause=similar_events.get("common_cause"),
            recommended_actions=actions,
            impact_assessment=impact,
            escalation=escalation,
            summary=summary,
            detailed_analysis=detailed,
        )
    
    def _get_similar_events_history(self, sensor_id: int, severity: str) -> dict:
        """Obtiene historial de eventos similares para el sensor."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT 
                        COUNT(*) as event_count,
                        MAX(created_at) as last_event_at,
                        AVG(DATEDIFF(minute, created_at, COALESCE(resolved_at, GETDATE()))) as avg_resolution_minutes
                    FROM dbo.ml_events
                    WHERE sensor_id = :sensor_id
                      AND event_type = :severity
                      AND created_at >= DATEADD(day, -30, GETDATE())
                """),
                {"sensor_id": sensor_id, "severity": severity.lower()},
            ).fetchone()
            
            if row and row.event_count > 0:
                return {
                    "count": int(row.event_count),
                    "last_at": row.last_event_at,
                    "avg_resolution_minutes": int(row.avg_resolution_minutes) if row.avg_resolution_minutes else None,
                    "common_cause": None,  # Se puede mejorar con análisis de payloads
                }
        except Exception:
            pass
        
        return {"count": 0, "last_at": None, "avg_resolution_minutes": None, "common_cause": None}
    
    def _generate_recommended_actions(
        self,
        *,
        sensor_type: str,
        location: str,
        severity: str,
        trend: str,
        anomaly_detected: bool,
    ) -> list[RecommendedAction]:
        """Genera acciones recomendadas específicas según el contexto."""
        actions = []
        sev = severity.lower()
        
        if sev == "critical":
            # Acción inmediata
            actions.append(RecommendedAction(
                action=f"Verificar estado físico del sensor {sensor_type} en {location}",
                urgency=ActionUrgency.IMMEDIATE,
                assigned_role="técnico_campo",
                estimated_time_minutes=15,
                equipment_needed=["multímetro", "herramientas básicas"],
                preconditions=["Notificar a operaciones antes de intervenir"],
            ))
            
            # Acción de contingencia
            if sensor_type == "temperature":
                actions.append(RecommendedAction(
                    action="Verificar sistema de climatización y ventilación",
                    urgency=ActionUrgency.IMMEDIATE,
                    assigned_role="técnico_hvac",
                    estimated_time_minutes=30,
                    equipment_needed=["termómetro de referencia"],
                ))
            elif sensor_type in {"power", "voltage"}:
                actions.append(RecommendedAction(
                    action="Revisar conexiones eléctricas y protecciones",
                    urgency=ActionUrgency.IMMEDIATE,
                    assigned_role="técnico_eléctrico",
                    estimated_time_minutes=20,
                    equipment_needed=["multímetro", "pinza amperimétrica"],
                    preconditions=["Usar EPP eléctrico"],
                ))
        
        elif sev == "warning":
            actions.append(RecommendedAction(
                action=f"Programar inspección de {sensor_type} en {location}",
                urgency=ActionUrgency.PRIORITY,
                assigned_role="técnico_mantenimiento",
                estimated_time_minutes=30,
            ))
            
            if anomaly_detected:
                actions.append(RecommendedAction(
                    action="Revisar calibración del sensor",
                    urgency=ActionUrgency.SCHEDULED,
                    assigned_role="técnico_instrumentación",
                    estimated_time_minutes=45,
                    equipment_needed=["patrón de calibración"],
                ))
        
        else:  # info
            actions.append(RecommendedAction(
                action="Continuar monitoreo normal",
                urgency=ActionUrgency.MONITOR,
                assigned_role="operador",
                estimated_time_minutes=0,
            ))
        
        return actions
    
    def _assess_impact(
        self,
        *,
        sensor_type: str,
        severity: str,
        trend: str,
        predicted_value: float,
        current_value: float,
    ) -> ImpactAssessment:
        """Evalúa el impacto potencial si no se actúa."""
        sev = severity.lower()
        
        if sev == "critical":
            affected = []
            damage = None
            time_to_impact = 30  # minutos
            
            if sensor_type == "temperature":
                affected = ["equipos de cómputo", "sistema de climatización"]
                damage = "Posible daño por sobrecalentamiento de equipos"
                time_to_impact = 60
            elif sensor_type in {"power", "voltage"}:
                affected = ["sistema eléctrico", "equipos conectados"]
                damage = "Posible daño a equipos por sobretensión o corte"
                time_to_impact = 15
            elif sensor_type == "humidity":
                affected = ["equipos electrónicos", "infraestructura"]
                damage = "Posible corrosión o condensación en equipos"
                time_to_impact = 120
            
            return ImpactAssessment(
                level=ImpactLevel.CRITICAL,
                description=f"Riesgo crítico detectado. {damage or 'Requiere atención inmediata.'}",
                time_to_impact_minutes=time_to_impact,
                affected_systems=affected,
                potential_damage=damage,
            )
        
        elif sev == "warning":
            return ImpactAssessment(
                level=ImpactLevel.MEDIUM,
                description="Degradación potencial si la tendencia continúa",
                time_to_impact_minutes=120,
                affected_systems=[sensor_type],
            )
        
        return ImpactAssessment(
            level=ImpactLevel.NONE,
            description="Sin impacto esperado en condiciones actuales",
            time_to_impact_minutes=None,
        )
    
    def _determine_escalation(
        self,
        *,
        severity: str,
        similar_events_count: int,
        anomaly_detected: bool,
    ) -> EscalationInfo:
        """Determina si se debe escalar y a quién."""
        sev = severity.lower()
        
        if sev == "critical":
            return EscalationInfo(
                should_escalate=True,
                escalation_level=2,
                reason="Condición crítica requiere atención de supervisión",
                notify_roles=["supervisor_turno", "jefe_mantenimiento"],
            )
        
        if sev == "warning" and similar_events_count >= 3:
            return EscalationInfo(
                should_escalate=True,
                escalation_level=1,
                reason=f"Evento recurrente ({similar_events_count} en últimos 30 días)",
                notify_roles=["supervisor_turno"],
            )
        
        if anomaly_detected and similar_events_count >= 5:
            return EscalationInfo(
                should_escalate=True,
                escalation_level=1,
                reason="Patrón anómalo recurrente requiere análisis",
                notify_roles=["analista_datos", "supervisor_turno"],
            )
        
        return EscalationInfo(
            should_escalate=False,
            escalation_level=0,
            reason="No requiere escalamiento",
        )
    
    def _generate_summary(
        self,
        *,
        sensor_type: str,
        location: str,
        severity: str,
        trend: str,
        predicted_value: float,
        similar_events_count: int,
    ) -> str:
        """Genera un resumen ejecutivo de la situación."""
        sev = severity.lower()
        
        trend_desc = {
            "rising": "tendencia ascendente",
            "falling": "tendencia descendente",
            "stable": "comportamiento estable",
        }.get(trend, trend)
        
        if sev == "critical":
            base = f"⚠️ CRÍTICO: Sensor {sensor_type} en {location} muestra {trend_desc}."
            if similar_events_count > 0:
                base += f" Este es un evento recurrente ({similar_events_count} similares en 30 días)."
            base += " Requiere atención inmediata."
        elif sev == "warning":
            base = f"⚡ ADVERTENCIA: {sensor_type} en {location} con {trend_desc}."
            base += " Se recomienda inspección prioritaria."
        else:
            base = f"✓ NORMAL: {sensor_type} en {location} operando dentro de parámetros."
        
        return base
    
    def _generate_detailed_analysis(
        self,
        *,
        sensor_type: str,
        location: str,
        predicted_value: float,
        current_value: float,
        trend: str,
        confidence: float,
        anomaly_score: float,
        similar_events: dict,
    ) -> str:
        """Genera un análisis técnico detallado."""
        delta = predicted_value - current_value
        delta_pct = (delta / current_value * 100) if current_value != 0 else 0
        
        parts = [
            f"Análisis de {sensor_type} en {location}:",
            f"- Valor actual: {current_value:.2f}",
            f"- Valor predicho: {predicted_value:.2f} (Δ {delta:+.2f}, {delta_pct:+.1f}%)",
            f"- Tendencia: {trend}",
            f"- Confianza del modelo: {confidence:.1%}",
        ]
        
        if anomaly_score > 0:
            parts.append(f"- Score de anomalía: {anomaly_score:.3f}")
        
        if similar_events.get("count", 0) > 0:
            parts.append(f"- Eventos similares (30 días): {similar_events['count']}")
            if similar_events.get("avg_resolution_minutes"):
                parts.append(f"- Tiempo promedio de resolución: {similar_events['avg_resolution_minutes']} min")
        
        return "\n".join(parts)
