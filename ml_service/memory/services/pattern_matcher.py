"""Pattern matcher service for decision memory.

Extracted from decision_memory.py for modularity.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ..models.decision_models import PatternMatch

logger = logging.getLogger(__name__)


class PatternMatcher:
    """Busca patrones similares en el historial de decisiones."""
    
    def __init__(self, conn: Connection):
        self._conn = conn
    
    def find_similar_patterns(
        self,
        pattern_signature: str,
        sensor_type: str,
        days_back: int = 30,
    ) -> list[PatternMatch]:
        """Busca patrones similares en el historial."""
        try:
            rows = self._conn.execute(
                text("""
                    SELECT 
                        pattern_signature,
                        COUNT(*) AS match_count,
                        AVG(resolution_time_minutes) AS avg_resolution_time,
                        STRING_AGG(DISTINCT actions_taken, ',') AS all_actions,
                        STRING_AGG(DISTINCT root_cause_identified, ',') AS all_causes,
                        AVG(CASE WHEN was_effective = 1 THEN 1.0 ELSE 0.0 END) AS success_rate,
                        MAX(created_at) AS last_match
                    FROM dbo.ml_decisions
                    WHERE pattern_signature = :pattern_signature
                      AND created_at >= DATEADD(day, -:days, GETDATE())
                    GROUP BY pattern_signature
                """),
                {"pattern_signature": pattern_signature, "days": days_back},
            ).fetchall()
            
            matches = []
            for row in rows:
                # Calcular días desde el último match
                last_match_days = None
                if row.last_match:
                    last_match_days = (datetime.now() - row.last_match).days
                
                # Procesar acciones efectivas
                effective_actions = []
                if row.all_actions:
                    # Contar frecuencia de acciones y tomar las más comunes
                    action_counts = {}
                    for action in row.all_actions.split(','):
                        action = action.strip()
                        if action:
                            action_counts[action] = action_counts.get(action, 0) + 1
                    
                    # Tomar las 3 acciones más efectivas
                    effective_actions = [
                        action for action, count in 
                        sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    ]
                
                # Causa más común
                common_cause = None
                if row.all_causes:
                    cause_counts = {}
                    for cause in row.all_causes.split(','):
                        cause = cause.strip()
                        if cause and cause.lower() != 'null':
                            cause_counts[cause] = cause_counts.get(cause, 0) + 1
                    
                    if cause_counts:
                        common_cause = max(cause_counts, key=cause_counts.get)
                
                matches.append(PatternMatch(
                    pattern_signature=row.pattern_signature,
                    match_count=row.match_count,
                    avg_resolution_time_minutes=float(row.avg_resolution_time) if row.avg_resolution_time else None,
                    effective_actions=effective_actions,
                    common_root_cause=common_cause,
                    success_rate=float(row.success_rate) if row.success_rate else 0.0,
                    last_match_days_ago=last_match_days,
                ))
            
            return matches
            
        except Exception as e:
            logger.warning("Failed to find similar patterns: %s", str(e))
            return []
    
    def get_pattern_statistics(
        self,
        sensor_type: Optional[str] = None,
        days_back: int = 30,
    ) -> dict:
        """Obtiene estadísticas generales de patrones."""
        try:
            where_clause = ""
            params = {"days": days_back}
            
            if sensor_type:
                where_clause = "AND sensor_type = :sensor_type"
                params["sensor_type"] = sensor_type
            
            rows = self._conn.execute(
                text(f"""
                    SELECT 
                        COUNT(*) AS total_decisions,
                        COUNT(DISTINCT pattern_signature) AS unique_patterns,
                        AVG(resolution_time_minutes) AS avg_resolution_time,
                        AVG(CASE WHEN was_effective = 1 THEN 1.0 ELSE 0.0 END) AS success_rate,
                        COUNT(CASE WHEN resolution_status = 'recurring' THEN 1 END) AS recurring_count
                    FROM dbo.ml_decisions
                    WHERE created_at >= DATEADD(day, -:days, GETDATE())
                      {where_clause}
                """),
                params,
            ).fetchone()
            
            if row:
                return {
                    "total_decisions": row.total_decisions,
                    "unique_patterns": row.unique_patterns,
                    "avg_resolution_time_minutes": float(row.avg_resolution_time) if row.avg_resolution_time else None,
                    "success_rate": float(row.success_rate) if row.success_rate else 0.0,
                    "recurring_rate": (row.recurring_count / row.total_decisions) if row.total_decisions > 0 else 0.0,
                }
            
        except Exception as e:
            logger.warning("Failed to get pattern statistics: %s", str(e))
        
        return {}
