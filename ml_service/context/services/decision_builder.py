"""Decision context builder service.

Extracted from decision_context.py for modularity.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ..models.decision_models import (
    DecisionContext,
    RecommendedAction,
    ImpactAssessment,
    EscalationInfo,
    ActionUrgency,
    ImpactLevel,
)

logger = logging.getLogger(__name__)


class DecisionContextBuilder:
    """Builder para construir DecisionContext."""
    
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
        """Construye el contexto de decisión completo."""
        
        # Generar acciones recomendadas
        actions = self._generate_actions(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            location=location,
            severity=severity,
            trend=trend,
            anomaly_detected=anomaly_detected,
        )
        
        # Evaluar impacto
        impact = self._assess_impact(
            device_id=device_id,
            severity=severity,
            sensor_type=sensor_type,
            trend=trend,
        )
        
        # Determinar escalamiento
        escalation = self._determine_escalation(
            severity=severity,
            impact=impact,
            anomaly_detected=anomaly_detected,
        )
        
        # Generar resumen
        summary = self._generate_summary(
            sensor_type=sensor_type,
            location=location,
            severity=severity,
            anomaly_detected=anomaly_detected,
            trend=trend,
        )
        
        # Generar análisis detallado
        detailed_analysis = self._generate_detailed_analysis(
            current_value=current_value,
            predicted_value=predicted_value,
            trend=trend,
            confidence=confidence,
            anomaly_score=anomaly_score,
            actions=actions,
            impact=impact,
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
            severity=severity,
            risk_level=risk_level,
            anomaly_detected=anomaly_detected,
            anomaly_score=anomaly_score,
            recommended_actions=actions,
            impact_assessment=impact,
            escalation=escalation,
            summary=summary,
            detailed_analysis=detailed_analysis,
            generated_at=datetime.now(timezone.utc),
        )
    
    def _generate_actions(
        self,
        sensor_id: int,
        sensor_type: str,
        location: str,
        severity: str,
        trend: str,
        anomaly_detected: bool,
    ) -> list[RecommendedAction]:
        """Genera acciones recomendadas basadas en el contexto."""
        actions = []
        
        if severity.lower() == "critical":
            actions.append(RecommendedAction(
                action=f"Verificar físicamente el sensor {sensor_id} en {location}",
                urgency=ActionUrgency.IMMEDIATE,
                assigned_role="técnico de campo",
                estimated_time_minutes=30,
                equipment_needed=["multímetro", "herramientas básicas"],
                preconditions=["permiso de acceso al área"],
            ))
        
        if anomaly_detected:
            actions.append(RecommendedAction(
                action=f"Revisar calibración del sensor {sensor_type}",
                urgency=ActionUrgency.PRIORITY,
                assigned_role="técnico especializado",
                estimated_time_minutes=45,
                equipment_needed=["equipo de calibración"],
                preconditions=["sensor fuera de servicio"],
            ))
        
        if trend == "rising" and severity.lower() in ("warning", "critical"):
            actions.append(RecommendedAction(
                action="Monitorear tendencia ascendente",
                urgency=ActionUrgency.MONITOR,
                assigned_role="operador",
                estimated_time_minutes=5,
                equipment_needed=[],
                preconditions=["acceso a sistema de monitoreo"],
            ))
        
        return actions
    
    def _assess_impact(
        self,
        device_id: int,
        severity: str,
        sensor_type: str,
        trend: str,
    ) -> Optional[ImpactAssessment]:
        """Evalúa el impacto si no se actúa."""
        try:
            row = self._conn.execute(
                text("""
                    SELECT device_type, name
                    FROM dbo.devices
                    WHERE id = :device_id
                """),
                {"device_id": device_id},
            ).fetchone()
            
            if row:
                device_type = str(row.device_type or "").lower()
                device_name = str(row.name or "")
                
                # Heurística de impacto
                if severity.lower() == "critical":
                    level = ImpactLevel.CRITICAL
                    description = f"Posible falla crítica en {device_name}"
                    time_to_impact = 30
                elif severity.lower() == "warning":
                    level = ImpactLevel.HIGH
                    description = f"Degradación del servicio en {device_name}"
                    time_to_impact = 120
                else:
                    level = ImpactLevel.MEDIUM
                    description = f"Operación subóptima en {device_name}"
                    time_to_impact = 240
                
                return ImpactAssessment(
                    level=level,
                    description=description,
                    time_to_impact_minutes=time_to_impact,
                    affected_systems=[device_type],
                    potential_damage=None,
                )
        except Exception as e:
            logger.warning("Failed to assess impact: %s", str(e))
        
        return None
    
    def _determine_escalation(
        self,
        severity: str,
        impact: Optional[ImpactAssessment],
        anomaly_detected: bool,
    ) -> Optional[EscalationInfo]:
        """Determina si se requiere escalamiento."""
        if severity.lower() == "critical":
            return EscalationInfo(
                should_escalate=True,
                escalation_level=2,
                reason="Evento crítico detectado",
                notify_roles=["supervisor", "gerente de operaciones"],
            )
        
        if impact and impact.level == ImpactLevel.CRITICAL:
            return EscalationInfo(
                should_escalate=True,
                escalation_level=1,
                reason="Impacto crítico potencial",
                notify_roles=["supervisor"],
            )
        
        if anomaly_detected and severity.lower() == "warning":
            return EscalationInfo(
                should_escalate=False,
                escalation_level=0,
                reason="Anomalía detectada, monitoreo activo",
                notify_roles=[],
            )
        
        return None
    
    def _generate_summary(
        self,
        sensor_type: str,
        location: str,
        severity: str,
        anomaly_detected: bool,
        trend: str,
    ) -> str:
        """Genera un resumen ejecutivo."""
        anomaly_text = " con anomalía detectada" if anomaly_detected else ""
        trend_text = f" en tendencia {trend}" if trend != "stable" else ""
        
        return (
            f"Sensor {sensor_type} en {location}{trend_text}{anomaly_text} "
            f"requiere atención {severity.lower()}."
        )
    
    def _generate_detailed_analysis(
        self,
        current_value: float,
        predicted_value: float,
        trend: str,
        confidence: float,
        anomaly_score: float,
        actions: list[RecommendedAction],
        impact: Optional[ImpactAssessment],
    ) -> str:
        """Genera análisis detallado."""
        delta = predicted_value - current_value
        delta_pct = (delta / current_value * 100) if current_value != 0 else 0
        
        analysis_parts = [
            f"El valor actual es {current_value:.2f} con una predicción de {predicted_value:.2f} "
            f"(cambio de {delta:+.2f}, {delta_pct:+.1f}%).",
            f"La confianza del modelo es {confidence:.2f} y el score de anomalía es {anomaly_score:.2f}.",
        ]
        
        if actions:
            analysis_parts.append(f"Se recomiendan {len(actions)} acciones:")
            for i, action in enumerate(actions, 1):
                analysis_parts.append(f"  {i}. {action.action}")
        
        if impact:
            analysis_parts.append(
                f"Si no se actúa, el impacto esperado es {impact.level.value}: {impact.description}"
            )
        
        return " ".join(analysis_parts)
