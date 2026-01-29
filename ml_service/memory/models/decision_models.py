"""Decision memory models.

Extracted from decision_memory.py for modularity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any


@dataclass
class DecisionRecord:
    """Registro de una decisión tomada."""
    
    id: Optional[int]
    
    # Identificación del evento
    sensor_id: int
    device_id: int
    event_type: str
    event_code: str
    
    # Firma del patrón (para matching)
    pattern_signature: str
    
    # Contexto del evento
    sensor_type: str
    severity: str
    trend: str
    anomaly_score: float
    predicted_value: float
    
    # Decisión tomada
    decision_type: str  # 'automated', 'manual', 'escalated'
    actions_taken: list[str]
    
    # Resultado
    resolution_status: str  # 'resolved', 'recurring', 'escalated', 'pending'
    resolution_time_minutes: Optional[int]
    root_cause_identified: Optional[str]
    was_effective: Optional[bool]
    
    # Feedback
    user_feedback: Optional[str]  # 'helpful', 'not_helpful', 'incorrect'
    feedback_notes: Optional[str]
    
    # Timestamps
    created_at: datetime
    resolved_at: Optional[datetime]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sensor_id": self.sensor_id,
            "device_id": self.device_id,
            "event_type": self.event_type,
            "event_code": self.event_code,
            "pattern_signature": self.pattern_signature,
            "sensor_type": self.sensor_type,
            "severity": self.severity,
            "trend": self.trend,
            "anomaly_score": self.anomaly_score,
            "predicted_value": self.predicted_value,
            "decision_type": self.decision_type,
            "actions_taken": self.actions_taken,
            "resolution_status": self.resolution_status,
            "resolution_time_minutes": self.resolution_time_minutes,
            "root_cause_identified": self.root_cause_identified,
            "was_effective": self.was_effective,
            "user_feedback": self.user_feedback,
            "feedback_notes": self.feedback_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class PatternMatch:
    """Coincidencia de patrón encontrada en el historial."""
    
    pattern_signature: str
    match_count: int
    avg_resolution_time_minutes: Optional[float]
    effective_actions: list[str]
    common_root_cause: Optional[str]
    success_rate: float
    last_match_days_ago: Optional[int]
    
    def to_dict(self) -> dict:
        return {
            "pattern_signature": self.pattern_signature,
            "match_count": self.match_count,
            "avg_resolution_time_minutes": self.avg_resolution_time_minutes,
            "effective_actions": self.effective_actions,
            "common_root_cause": self.common_root_cause,
            "success_rate": self.success_rate,
            "last_match_days_ago": self.last_match_days_ago,
        }


@dataclass
class HistoricalInsight:
    """Insight histórico basado en decisiones pasadas."""
    
    has_history: bool
    similar_events_count: int
    is_recurring_issue: bool
    suggested_actions: list[str]
    suggested_root_cause: Optional[str]
    estimated_resolution_time_minutes: Optional[int]
    confidence: float
    pattern_matches: list[PatternMatch]
    
    def to_dict(self) -> dict:
        return {
            "has_history": self.has_history,
            "similar_events_count": self.similar_events_count,
            "is_recurring_issue": self.is_recurring_issue,
            "suggested_actions": self.suggested_actions,
            "suggested_root_cause": self.suggested_root_cause,
            "estimated_resolution_time_minutes": self.estimated_resolution_time_minutes,
            "confidence": self.confidence,
            "pattern_matches": [pm.to_dict() for pm in self.pattern_matches],
        }


def create_pattern_signature(
    sensor_type: str,
    severity: str,
    trend: str,
    anomaly_score: float,
    event_code: str,
) -> str:
    """Crea una firma única para el patrón de decisión."""
    # Normalizar valores para consistencia
    normalized_anomaly = round(anomaly_score, 2)
    normalized_trend = trend.lower().strip()
    
    # Crear hash del patrón
    pattern_data = {
        "sensor_type": sensor_type.lower().strip(),
        "severity": severity.lower().strip(),
        "trend": normalized_trend,
        "anomaly_score": normalized_anomaly,
        "event_code": event_code.lower().strip(),
    }
    
    pattern_str = json.dumps(pattern_data, sort_keys=True)
    return hashlib.md5(pattern_str.encode()).hexdigest()[:16]
