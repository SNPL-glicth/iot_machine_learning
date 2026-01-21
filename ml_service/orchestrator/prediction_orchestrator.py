"""Orquestador de Predicciones ML.

Este módulo integra todos los componentes de contexto para generar
predicciones enriquecidas con:
- Contexto de decisión accionable (Falencia 1)
- Correlación entre sensores (Falencia 2)
- Explicaciones contextuales (Falencia 3)
- Contexto operacional (Falencia 4)
- Memoria de decisiones (Falencia 5)
- Payload enriquecido para LLM (Falencia 6)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Any

from sqlalchemy.engine import Connection

from iot_machine_learning.ml_service.context.decision_context import (
    DecisionContext,
    DecisionContextBuilder,
)
from iot_machine_learning.ml_service.context.operational_context import (
    OperationalContext,
    OperationalContextBuilder,
    adjust_severity_with_context,
)
from iot_machine_learning.ml_service.correlation.sensor_correlator import (
    CorrelationResult,
    SensorCorrelator,
    correlate_sensor_with_device,
)
from iot_machine_learning.ml_service.explain.contextual_explainer import (
    EnrichedContext,
    ExplanationResult,
    ContextualExplainer,
    create_contextual_explanation,
)
from iot_machine_learning.ml_service.memory.decision_memory import (
    DecisionMemory,
    HistoricalInsight,
    get_historical_insight_for_event,
    record_ml_decision,
)
from iot_machine_learning.ml_service.repository.sensor_repository import (
    SensorMetadata,
    load_sensor_metadata,
    get_device_id_for_sensor,
)

logger = logging.getLogger(__name__)


@dataclass
class EnrichedPrediction:
    """Predicción enriquecida con todo el contexto disponible."""
    
    # Identificación
    sensor_id: int
    device_id: int
    
    # Predicción base
    predicted_value: float
    current_value: float
    trend: str
    confidence: float
    anomaly_score: float
    is_anomaly: bool
    
    # Severidad (ajustada por contexto operacional)
    base_severity: str
    adjusted_severity: str
    risk_level: str
    
    # Contextos
    decision_context: Optional[DecisionContext]
    operational_context: Optional[OperationalContext]
    correlation_result: Optional[CorrelationResult]
    historical_insight: Optional[HistoricalInsight]
    explanation_result: Optional[ExplanationResult]
    
    # Payload para AI Explainer
    enriched_payload: dict
    
    # Metadata
    generated_at: datetime
    
    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "device_id": self.device_id,
            "predicted_value": self.predicted_value,
            "current_value": self.current_value,
            "trend": self.trend,
            "confidence": self.confidence,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "base_severity": self.base_severity,
            "adjusted_severity": self.adjusted_severity,
            "risk_level": self.risk_level,
            "decision_context": self.decision_context.to_dict() if self.decision_context else None,
            "operational_context": self.operational_context.to_dict() if self.operational_context else None,
            "correlation_result": self.correlation_result.to_dict() if self.correlation_result else None,
            "historical_insight": self.historical_insight.to_dict() if self.historical_insight else None,
            "explanation_result": self.explanation_result.to_dict() if self.explanation_result else None,
            "enriched_payload": self.enriched_payload,
            "generated_at": self.generated_at.isoformat(),
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)
    
    def get_explanation_json(self) -> str:
        """Genera el JSON de explicación para persistir en predictions.explanation."""
        
        explanation_data = {
            "source": "ml_orchestrator",
            "version": "2.0",
            "severity": self.adjusted_severity.upper(),
            "risk_level": self.risk_level,
        }
        
        # Agregar resumen del contexto de decisión
        if self.decision_context:
            explanation_data["summary"] = self.decision_context.summary
            explanation_data["detailed_analysis"] = self.decision_context.detailed_analysis
            explanation_data["recommended_actions"] = [
                a.to_dict() for a in self.decision_context.recommended_actions
            ]
            if self.decision_context.impact_assessment:
                explanation_data["impact"] = self.decision_context.impact_assessment.to_dict()
            if self.decision_context.escalation:
                explanation_data["escalation"] = self.decision_context.escalation.to_dict()
        
        # Agregar correlaciones
        if self.correlation_result and self.correlation_result.is_significant:
            explanation_data["correlation"] = {
                "pattern": self.correlation_result.pattern_detected.value if self.correlation_result.pattern_detected else None,
                "confidence": self.correlation_result.pattern_confidence,
                "description": self.correlation_result.description,
                "root_cause_hypothesis": self.correlation_result.root_cause_hypothesis,
            }
        
        # Agregar insights históricos
        if self.historical_insight and self.historical_insight.has_history:
            explanation_data["historical"] = {
                "similar_events": self.historical_insight.similar_events_count,
                "is_recurring": self.historical_insight.is_recurring_issue,
                "suggested_actions": self.historical_insight.suggested_actions,
                "suggested_root_cause": self.historical_insight.suggested_root_cause,
                "estimated_resolution_time": self.historical_insight.estimated_resolution_time,
            }
        
        # Agregar explicación contextual
        if self.explanation_result:
            explanation_data["explanation"] = self.explanation_result.explanation
            explanation_data["possible_causes"] = self.explanation_result.possible_causes
            explanation_data["recommended_action"] = self.explanation_result.recommended_action
            explanation_data["explanation_confidence"] = self.explanation_result.confidence
            explanation_data["explanation_source"] = self.explanation_result.source
        
        # Agregar contexto operacional
        if self.operational_context:
            explanation_data["operational"] = {
                "work_shift": self.operational_context.work_shift.value,
                "staff_availability": self.operational_context.staff_availability.value,
                "response_time_minutes": self.operational_context.response_time_minutes,
                "severity_multiplier": self.operational_context.severity_multiplier,
                "urgency_boost": self.operational_context.urgency_boost,
            }
        
        return json.dumps(explanation_data, ensure_ascii=False)


class PredictionOrchestrator:
    """Orquestador que integra todos los módulos de contexto.
    
    Este es el punto de entrada principal para generar predicciones
    enriquecidas con todo el contexto disponible.
    """
    
    def __init__(self, conn: Connection):
        self._conn = conn
        self._decision_builder = DecisionContextBuilder(conn)
        self._operational_builder = OperationalContextBuilder(conn)
        self._correlator = SensorCorrelator(conn)
        self._memory = DecisionMemory(conn)
        self._explainer = ContextualExplainer(conn)
    
    def enrich_prediction(
        self,
        *,
        sensor_id: int,
        predicted_value: float,
        current_value: float,
        trend: str,
        confidence: float,
        anomaly_score: float,
        is_anomaly: bool,
        base_severity: str,
        risk_level: str,
        horizon_minutes: int = 10,
    ) -> EnrichedPrediction:
        """Enriquece una predicción con todo el contexto disponible.
        
        Este método:
        1. Carga metadata del sensor
        2. Obtiene correlaciones con otros sensores
        3. Consulta historial de decisiones similares
        4. Ajusta severidad por contexto operacional
        5. Genera contexto de decisión accionable
        6. Crea explicación contextual
        7. Registra la decisión en memoria
        """
        
        # 1. Cargar metadata del sensor
        try:
            sensor_meta = load_sensor_metadata(self._conn, sensor_id)
            device_id = get_device_id_for_sensor(self._conn, sensor_id)
        except Exception as e:
            logger.warning(
                "[ORCHESTRATOR] Failed to load sensor metadata for sensor_id=%s: %s",
                sensor_id, str(e)
            )
            sensor_meta = SensorMetadata(
                sensor_id=sensor_id,
                sensor_type="unknown",
                unit="",
                location="unknown",
                criticality="medium",
            )
            device_id = 0
        
        # 2. Obtener correlaciones
        correlation_result = None
        try:
            correlation_result = self._correlator.analyze_all_devices_for_sensor(sensor_id)
        except Exception as e:
            logger.warning(
                "[ORCHESTRATOR] Failed to analyze correlations for sensor_id=%s: %s",
                sensor_id, str(e)
            )
        
        # 3. Consultar historial de decisiones similares
        historical_insight = None
        try:
            event_code = "ANOMALY_DETECTED" if is_anomaly else "PRED_THRESHOLD_BREACH"
            historical_insight = self._memory.get_historical_insight(
                sensor_type=sensor_meta.sensor_type,
                severity=base_severity,
                trend=trend,
                event_code=event_code,
            )
        except Exception as e:
            logger.warning(
                "[ORCHESTRATOR] Failed to get historical insight for sensor_id=%s: %s",
                sensor_id, str(e)
            )
        
        # 4. Ajustar severidad por contexto operacional
        adjusted_severity = base_severity
        operational_context = None
        try:
            adjusted_severity, operational_context = adjust_severity_with_context(
                self._conn,
                sensor_id=sensor_id,
                device_id=device_id,
                base_severity=base_severity,
            )
        except Exception as e:
            logger.warning(
                "[ORCHESTRATOR] Failed to adjust severity for sensor_id=%s: %s",
                sensor_id, str(e)
            )
        
        # 5. Generar contexto de decisión accionable
        decision_context = None
        try:
            decision_context = self._decision_builder.build(
                sensor_id=sensor_id,
                device_id=device_id,
                sensor_type=sensor_meta.sensor_type,
                location=sensor_meta.location,
                predicted_value=predicted_value,
                current_value=current_value,
                trend=trend,
                confidence=confidence,
                prediction_horizon_minutes=horizon_minutes,
                severity=adjusted_severity,
                risk_level=risk_level,
                anomaly_detected=is_anomaly,
                anomaly_score=anomaly_score,
            )
        except Exception as e:
            logger.warning(
                "[ORCHESTRATOR] Failed to build decision context for sensor_id=%s: %s",
                sensor_id, str(e)
            )
        
        # 6. Crear explicación contextual
        explanation_result = None
        try:
            explanation_result = create_contextual_explanation(
                self._conn,
                sensor_id=sensor_id,
                predicted_value=predicted_value,
                trend=trend,
                confidence=confidence,
                horizon_minutes=horizon_minutes,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
            )
        except Exception as e:
            logger.warning(
                "[ORCHESTRATOR] Failed to create explanation for sensor_id=%s: %s",
                sensor_id, str(e)
            )
        
        # 7. Construir payload enriquecido para AI Explainer
        enriched_payload = self._build_enriched_payload(
            sensor_meta=sensor_meta,
            device_id=device_id,
            predicted_value=predicted_value,
            current_value=current_value,
            trend=trend,
            confidence=confidence,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            severity=adjusted_severity,
            risk_level=risk_level,
            horizon_minutes=horizon_minutes,
            historical_insight=historical_insight,
            correlation_result=correlation_result,
            operational_context=operational_context,
        )
        
        # 8. Registrar decisión en memoria (si hay evento significativo)
        if adjusted_severity.lower() in ("warning", "critical") or is_anomaly:
            try:
                actions = []
                if decision_context and decision_context.recommended_actions:
                    actions = [a.action for a in decision_context.recommended_actions]
                
                record_ml_decision(
                    self._conn,
                    sensor_id=sensor_id,
                    device_id=device_id,
                    event_type=adjusted_severity.lower(),
                    event_code="ANOMALY_DETECTED" if is_anomaly else "PRED_THRESHOLD_BREACH",
                    sensor_type=sensor_meta.sensor_type,
                    severity=adjusted_severity,
                    trend=trend,
                    anomaly_score=anomaly_score,
                    predicted_value=predicted_value,
                    actions_taken=actions,
                )
            except Exception as e:
                logger.warning(
                    "[ORCHESTRATOR] Failed to record decision for sensor_id=%s: %s",
                    sensor_id, str(e)
                )
        
        return EnrichedPrediction(
            sensor_id=sensor_id,
            device_id=device_id,
            predicted_value=predicted_value,
            current_value=current_value,
            trend=trend,
            confidence=confidence,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            base_severity=base_severity,
            adjusted_severity=adjusted_severity,
            risk_level=risk_level,
            decision_context=decision_context,
            operational_context=operational_context,
            correlation_result=correlation_result,
            historical_insight=historical_insight,
            explanation_result=explanation_result,
            enriched_payload=enriched_payload,
            generated_at=datetime.now(timezone.utc),
        )
    
    def _build_enriched_payload(
        self,
        *,
        sensor_meta: SensorMetadata,
        device_id: int,
        predicted_value: float,
        current_value: float,
        trend: str,
        confidence: float,
        anomaly_score: float,
        is_anomaly: bool,
        severity: str,
        risk_level: str,
        horizon_minutes: int,
        historical_insight: Optional[HistoricalInsight],
        correlation_result: Optional[CorrelationResult],
        operational_context: Optional[OperationalContext],
    ) -> dict:
        """Construye el payload enriquecido para el AI Explainer v2."""
        
        payload = {
            "asset_info": {
                "asset_id": f"device_{device_id}",
                "asset_type": "iot_device",
                "location": sensor_meta.location,
            },
            "sensor_info": {
                "sensor_id": sensor_meta.sensor_id,
                "sensor_type": sensor_meta.sensor_type,
                "sensor_name": f"Sensor {sensor_meta.sensor_id}",
                "unit": sensor_meta.unit,
                "physical_min": None,
                "physical_max": None,
                "operational_range": {"min": None, "max": None},
            },
            "device_info": {
                "device_id": device_id,
                "device_name": sensor_meta.location,
                "device_type": "",
            },
            "model_output": {
                "current_value": current_value,
                "predicted_value": predicted_value,
                "trend": trend,
                "confidence": confidence,
                "anomaly_score": anomaly_score,
                "horizon_minutes": horizon_minutes,
            },
            "decision_output": {
                "severity": severity.upper(),
                "cause": "statistical_anomaly" if is_anomaly else "physical_violation",
                "state": "OUT_OF_RANGE" if severity.lower() == "critical" else "IN_RANGE",
                "risk_level": risk_level,
            },
            "historical_context": {
                "recent_anomalies": 0,
                "maintenance_due": False,
                "stable_minutes": 0,
                "similar_events_count": 0,
                "last_similar_event_days_ago": None,
                "common_root_cause": None,
                "avg_resolution_time_minutes": None,
                "effective_actions": [],
            },
            "correlation_context": {
                "has_correlations": False,
                "pattern_detected": None,
                "pattern_confidence": 0.0,
                "correlated_sensors": [],
                "root_cause_hypothesis": None,
            },
            "operational_context": {
                "work_shift": "morning",
                "is_business_hours": True,
                "staff_availability": "full",
                "response_time_minutes": 15,
                "production_impact": "none",
            },
        }
        
        # Enriquecer con historial
        if historical_insight:
            payload["historical_context"].update({
                "similar_events_count": historical_insight.similar_events_count,
                "common_root_cause": historical_insight.suggested_root_cause,
                "avg_resolution_time_minutes": historical_insight.estimated_resolution_time,
                "effective_actions": historical_insight.suggested_actions,
            })
        
        # Enriquecer con correlaciones
        if correlation_result:
            payload["correlation_context"].update({
                "has_correlations": correlation_result.is_significant,
                "pattern_detected": correlation_result.pattern_detected.value if correlation_result.pattern_detected else None,
                "pattern_confidence": correlation_result.pattern_confidence,
                "correlated_sensors": [
                    {
                        "sensor_type": s.sensor_type,
                        "trend": s.trend,
                        "severity": s.severity,
                    }
                    for s in correlation_result.correlated_sensors
                ],
                "root_cause_hypothesis": correlation_result.root_cause_hypothesis,
            })
        
        # Enriquecer con contexto operacional
        if operational_context:
            payload["operational_context"].update({
                "work_shift": operational_context.work_shift.value,
                "is_business_hours": operational_context.is_business_hours,
                "staff_availability": operational_context.staff_availability.value,
                "response_time_minutes": operational_context.response_time_minutes,
                "production_impact": operational_context.production_impact.value,
            })
        
        return payload
    
    def clear_caches(self) -> None:
        """Limpia los caches de los componentes."""
        self._correlator.clear_cache()


def enrich_prediction_with_context(
    conn: Connection,
    *,
    sensor_id: int,
    predicted_value: float,
    current_value: float,
    trend: str,
    confidence: float,
    anomaly_score: float,
    is_anomaly: bool,
    base_severity: str,
    risk_level: str,
    horizon_minutes: int = 10,
) -> EnrichedPrediction:
    """Función de conveniencia para enriquecer una predicción.
    
    Esta función es el punto de entrada principal para el batch runner.
    """
    orchestrator = PredictionOrchestrator(conn)
    return orchestrator.enrich_prediction(
        sensor_id=sensor_id,
        predicted_value=predicted_value,
        current_value=current_value,
        trend=trend,
        confidence=confidence,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        base_severity=base_severity,
        risk_level=risk_level,
        horizon_minutes=horizon_minutes,
    )
