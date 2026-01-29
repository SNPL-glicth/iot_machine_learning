"""Decision memory service.

Extracted from decision_memory.py for modularity.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ..models.decision_models import (
    DecisionRecord,
    HistoricalInsight,
    create_pattern_signature,
)

logger = logging.getLogger(__name__)


class DecisionMemoryService:
    """Servicio para gestionar la memoria de decisiones."""
    
    def __init__(self, conn: Connection):
        self._conn = conn
    
    def record_decision(
        self,
        *,
        sensor_id: int,
        device_id: int,
        event_type: str,
        event_code: str,
        sensor_type: str,
        severity: str,
        trend: str,
        anomaly_score: float,
        predicted_value: float,
        actions_taken: list[str],
    ) -> int:
        """Registra una decisión tomada."""
        
        # Crear firma del patrón
        pattern_signature = create_pattern_signature(
            sensor_type=sensor_type,
            severity=severity,
            trend=trend,
            anomaly_score=anomaly_score,
            event_code=event_code,
        )
        
        # Insertar registro
        row = self._conn.execute(
            text("""
                INSERT INTO dbo.ml_decisions (
                  sensor_id, device_id, event_type, event_code,
                  pattern_signature, sensor_type, severity, trend,
                  anomaly_score, predicted_value, decision_type,
                  actions_taken, created_at
                )
                OUTPUT INSERTED.id
                VALUES (
                  :sensor_id, :device_id, :event_type, :event_code,
                  :pattern_signature, :sensor_type, :severity, :trend,
                  :anomaly_score, :predicted_value, 'automated',
                  :actions_taken, GETDATE()
                )
            """),
            {
                "sensor_id": sensor_id,
                "device_id": device_id,
                "event_type": event_type,
                "event_code": event_code,
                "pattern_signature": pattern_signature,
                "sensor_type": sensor_type,
                "severity": severity,
                "trend": trend,
                "anomaly_score": anomaly_score,
                "predicted_value": predicted_value,
                "actions_taken": ",".join(actions_taken),
            },
        ).fetchone()
        
        if not row:
            raise RuntimeError("Failed to record decision")
        
        return int(row[0])
    
    def get_historical_insight(
        self,
        sensor_type: str,
        severity: str,
        trend: str,
        event_code: str,
        anomaly_score: float,
        days_back: int = 30,
    ) -> HistoricalInsight:
        """Obtiene insight histórico basado en decisiones pasadas."""
        
        pattern_signature = create_pattern_signature(
            sensor_type=sensor_type,
            severity=severity,
            trend=trend,
            anomaly_score=anomaly_score,
            event_code=event_code,
        )
        
        try:
            # Buscar patrones similares
            from .pattern_matcher import PatternMatcher
            matcher = PatternMatcher(self._conn)
            pattern_matches = matcher.find_similar_patterns(pattern_signature, sensor_type, days_back)
            
            if not pattern_matches:
                return HistoricalInsight(
                    has_history=False,
                    similar_events_count=0,
                    is_recurring_issue=False,
                    suggested_actions=[],
                    suggested_root_cause=None,
                    estimated_resolution_time_minutes=None,
                    confidence=0.0,
                    pattern_matches=[],
                )
            
            # Calcular insight basado en los matches
            total_matches = sum(pm.match_count for pm in pattern_matches)
            avg_resolution = sum(
                pm.avg_resolution_time_minutes or 0 for pm in pattern_matches
            ) / len(pattern_matches)
            
            # Acciones más efectivas
            all_actions = []
            for pm in pattern_matches:
                all_actions.extend(pm.effective_actions)
            
            # Contar frecuencia de acciones
            action_counts = {}
            for action in all_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            suggested_actions = [
                action for action, count in 
                sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            ]
            
            # Causa más común
            common_causes = [pm.common_root_cause for pm in pattern_matches if pm.common_root_cause]
            suggested_root_cause = max(set(common_causes), key=common_causes.count) if common_causes else None
            
            # Determinar si es recurrente
            is_recurring = total_matches >= 3
            
            # Calcular confianza
            confidence = min(0.9, total_matches / 10.0)  # Más matches = más confianza
            
            return HistoricalInsight(
                has_history=True,
                similar_events_count=total_matches,
                is_recurring_issue=is_recurring,
                suggested_actions=suggested_actions,
                suggested_root_cause=suggested_root_cause,
                estimated_resolution_time_minutes=int(avg_resolution) if avg_resolution > 0 else None,
                confidence=confidence,
                pattern_matches=pattern_matches,
            )
            
        except Exception as e:
            logger.warning("Failed to get historical insight: %s", str(e))
            return HistoricalInsight(
                has_history=False,
                similar_events_count=0,
                is_recurring_issue=False,
                suggested_actions=[],
                suggested_root_cause=None,
                estimated_resolution_time_minutes=None,
                confidence=0.0,
                pattern_matches=[],
            )
    
    def update_decision_feedback(
        self,
        decision_id: int,
        user_feedback: str,
        feedback_notes: Optional[str] = None,
    ) -> bool:
        """Actualiza el feedback de una decisión."""
        try:
            self._conn.execute(
                text("""
                    UPDATE dbo.ml_decisions
                    SET user_feedback = :user_feedback,
                        feedback_notes = :feedback_notes,
                        updated_at = GETDATE()
                    WHERE id = :decision_id
                """),
                {
                    "decision_id": decision_id,
                    "user_feedback": user_feedback,
                    "feedback_notes": feedback_notes,
                },
            )
            return True
        except Exception as e:
            logger.warning("Failed to update decision feedback: %s", str(e))
            return False
    
    def get_recent_decisions(
        self,
        sensor_id: Optional[int] = None,
        days_back: int = 7,
        limit: int = 50,
    ) -> list[DecisionRecord]:
        """Obtiene decisiones recientes."""
        try:
            where_clause = ""
            params = {"days": days_back, "limit": limit}
            
            if sensor_id:
                where_clause = "AND sensor_id = :sensor_id"
                params["sensor_id"] = sensor_id
            
            rows = self._conn.execute(
                text(f"""
                    SELECT TOP (:limit)
                        id, sensor_id, device_id, event_type, event_code,
                        pattern_signature, sensor_type, severity, trend,
                        anomaly_score, predicted_value, decision_type,
                        actions_taken, resolution_status, resolution_time_minutes,
                        root_cause_identified, was_effective, user_feedback,
                        feedback_notes, created_at, resolved_at
                    FROM dbo.ml_decisions
                    WHERE created_at >= DATEADD(day, -:days, GETDATE())
                      {where_clause}
                    ORDER BY created_at DESC
                """),
                params,
            ).fetchall()
            
            decisions = []
            for row in rows:
                # Parse actions_taken
                actions = []
                if row.actions_taken:
                    actions = [a.strip() for a in row.actions_taken.split(',') if a.strip()]
                
                decisions.append(DecisionRecord(
                    id=row.id,
                    sensor_id=row.sensor_id,
                    device_id=row.device_id,
                    event_type=row.event_type,
                    event_code=row.event_code,
                    pattern_signature=row.pattern_signature,
                    sensor_type=row.sensor_type,
                    severity=row.severity,
                    trend=row.trend,
                    anomaly_score=row.anomaly_score,
                    predicted_value=row.predicted_value,
                    decision_type=row.decision_type,
                    actions_taken=actions,
                    resolution_status=row.resolution_status,
                    resolution_time_minutes=row.resolution_time_minutes,
                    root_cause_identified=row.root_cause_identified,
                    was_effective=row.was_effective,
                    user_feedback=row.user_feedback,
                    feedback_notes=row.feedback_notes,
                    created_at=row.created_at,
                    resolved_at=row.resolved_at,
                ))
            
            return decisions
            
        except Exception as e:
            logger.warning("Failed to get recent decisions: %s", str(e))
            return []
