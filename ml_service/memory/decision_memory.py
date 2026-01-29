"""Módulo de Memoria de Decisiones para ML.

REFACTORIZADO 2026-01-29:
- Modelos extraídos a models/decision_models.py
- Servicios extraídos a services/
- Este archivo ahora es solo el punto de entrada (~50 líneas, antes 744)

Estructura:
- models/decision_models.py: Dataclasses (DecisionRecord, PatternMatch, etc.)
- services/memory_service.py: DecisionMemoryService
- services/pattern_matcher.py: PatternMatcher
"""

from __future__ import annotations

from sqlalchemy.engine import Connection

from .models.decision_models import (
    DecisionRecord,
    PatternMatch,
    HistoricalInsight,
    create_pattern_signature,
)
from .services import DecisionMemoryService, PatternMatcher

# Re-exportar para compatibilidad
__all__ = [
    "DecisionRecord",
    "PatternMatch",
    "HistoricalInsight",
    "create_pattern_signature",
    "DecisionMemoryService",
    "PatternMatcher",
]

# Clase principal para compatibilidad con código existente
class DecisionMemory:
    """Wrapper para compatibilidad con DecisionMemoryService."""
    
    def __init__(self, conn: Connection):
        self._service = DecisionMemoryService(conn)
    
    def get_historical_insight(
        self,
        sensor_type: str,
        severity: str,
        trend: str,
        event_code: str,
        anomaly_score: float,
        days_back: int = 30,
    ) -> HistoricalInsight:
        """Obtiene insight histórico."""
        return self._service.get_historical_insight(
            sensor_type=sensor_type,
            severity=severity,
            trend=trend,
            event_code=event_code,
            anomaly_score=anomaly_score,
            days_back=days_back,
        )


# Funciones de conveniencia para compatibilidad
def record_ml_decision(
    conn: Connection,
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
    """Registra una decisión ML."""
    service = DecisionMemoryService(conn)
    return service.record_decision(
        sensor_id=sensor_id,
        device_id=device_id,
        event_type=event_type,
        event_code=event_code,
        sensor_type=sensor_type,
        severity=severity,
        trend=trend,
        anomaly_score=anomaly_score,
        predicted_value=predicted_value,
        actions_taken=actions_taken,
    )


def get_historical_insight_for_event(
    conn: Connection,
    *,
    sensor_type: str,
    severity: str,
    trend: str,
    event_code: str,
    anomaly_score: float,
    days_back: int = 30,
) -> HistoricalInsight:
    """Obtiene insight histórico para un evento."""
    service = DecisionMemoryService(conn)
    return service.get_historical_insight(
        sensor_type=sensor_type,
        severity=severity,
        trend=trend,
        event_code=event_code,
        anomaly_score=anomaly_score,
        days_back=days_back,
    )
