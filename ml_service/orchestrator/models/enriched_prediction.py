"""Enriched prediction model for orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

from iot_machine_learning.ml_service.context.models import OperationalContext
from iot_machine_learning.ml_service.context.decision_context import DecisionContext
from iot_machine_learning.ml_service.correlation.sensor_correlator import CorrelationResult
from iot_machine_learning.ml_service.memory.decision_memory import HistoricalInsight
from iot_machine_learning.ml_service.explain.models import ExplanationResult


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
        
        if self.correlation_result and self.correlation_result.is_significant:
            explanation_data["correlation"] = {
                "pattern": self.correlation_result.pattern_detected.value if self.correlation_result.pattern_detected else None,
                "confidence": self.correlation_result.pattern_confidence,
                "description": self.correlation_result.description,
                "root_cause_hypothesis": self.correlation_result.root_cause_hypothesis,
            }
        
        if self.historical_insight and self.historical_insight.has_history:
            explanation_data["historical"] = {
                "similar_events": self.historical_insight.similar_events_count,
                "is_recurring": self.historical_insight.is_recurring_issue,
                "suggested_actions": self.historical_insight.suggested_actions,
                "suggested_root_cause": self.historical_insight.suggested_root_cause,
                "estimated_resolution_time": self.historical_insight.estimated_resolution_time,
            }
        
        if self.explanation_result:
            explanation_data["explanation"] = self.explanation_result.explanation
            explanation_data["possible_causes"] = self.explanation_result.possible_causes
            explanation_data["recommended_action"] = self.explanation_result.recommended_action
            explanation_data["explanation_confidence"] = self.explanation_result.confidence
            explanation_data["explanation_source"] = self.explanation_result.source
        
        if self.operational_context:
            explanation_data["operational"] = {
                "work_shift": self.operational_context.work_shift.value,
                "staff_availability": self.operational_context.staff_availability.value,
                "response_time_minutes": self.operational_context.response_time_minutes,
                "severity_multiplier": self.operational_context.severity_multiplier,
                "urgency_boost": self.operational_context.urgency_boost,
            }
        
        return json.dumps(explanation_data, ensure_ascii=False)
