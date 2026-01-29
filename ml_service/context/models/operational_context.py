"""Operational context dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .work_shift import WorkShift, StaffAvailability, ProductionImpact


@dataclass
class OperationalContext:
    """Contexto operacional para ajuste de severidad."""
    
    # Temporal
    current_time: datetime
    work_shift: WorkShift
    is_business_hours: bool
    is_weekend: bool
    is_holiday: bool
    
    # Personal
    staff_availability: StaffAvailability
    response_time_minutes: int  # Tiempo estimado de respuesta
    
    # Producción
    production_impact: ProductionImpact
    affected_processes: list[str]
    
    # Historial de mantenimiento
    days_since_last_maintenance: Optional[int]
    pending_maintenance_tasks: int
    recent_incidents_count: int  # Incidentes en últimos 7 días
    
    # Factores de ajuste
    severity_multiplier: float  # 0.5 - 2.0
    urgency_boost: bool  # Si se debe aumentar la urgencia
    
    def to_dict(self) -> dict:
        return {
            "temporal": {
                "current_time": self.current_time.isoformat(),
                "work_shift": self.work_shift.value,
                "is_business_hours": self.is_business_hours,
                "is_weekend": self.is_weekend,
                "is_holiday": self.is_holiday,
            },
            "staff": {
                "availability": self.staff_availability.value,
                "response_time_minutes": self.response_time_minutes,
            },
            "production": {
                "impact": self.production_impact.value,
                "affected_processes": self.affected_processes,
            },
            "maintenance": {
                "days_since_last": self.days_since_last_maintenance,
                "pending_tasks": self.pending_maintenance_tasks,
                "recent_incidents": self.recent_incidents_count,
            },
            "adjustments": {
                "severity_multiplier": self.severity_multiplier,
                "urgency_boost": self.urgency_boost,
            },
        }
