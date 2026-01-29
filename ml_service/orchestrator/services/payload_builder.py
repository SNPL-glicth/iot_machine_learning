"""Enriched payload builder service for orchestrator."""

from __future__ import annotations

from typing import Optional

from iot_machine_learning.ml_service.context.models import OperationalContext
from iot_machine_learning.ml_service.correlation.sensor_correlator import CorrelationResult
from iot_machine_learning.ml_service.memory.decision_memory import HistoricalInsight
from iot_machine_learning.ml_service.repository.sensor_repository import SensorMetadata


class EnrichedPayloadBuilder:
    """Builds enriched payload for AI Explainer."""
    
    def build(
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
