"""Módulo de Contexto Operacional para ML.

REFACTORIZADO 2026-01-29:
- Modelos extraídos a models/
- Servicios extraídos a services/
- Este archivo ahora es solo el orquestador (~100 líneas, antes 464)

Estructura:
- models/work_shift.py: Enums (WorkShift, StaffAvailability, ProductionImpact)
- models/operational_context.py: OperationalContext dataclass
- services/shift_calculator.py: Cálculo de turnos
- services/impact_assessor.py: Evaluación de impacto
- services/severity_calculator.py: Cálculo de severidad
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.engine import Connection

from .models import (
    WorkShift,
    StaffAvailability,
    ProductionImpact,
    OperationalContext,
)
from .services import (
    ShiftCalculator,
    ProductionImpactAssessor,
    SeverityCalculator,
)

logger = logging.getLogger(__name__)


class OperationalContextBuilder:
    """Builder para construir OperationalContext."""
    
    def __init__(self, conn: Connection):
        self._conn = conn
        self._shift_calc = ShiftCalculator()
        self._impact_assessor = ProductionImpactAssessor(conn)
        self._severity_calc = SeverityCalculator()
    
    def build(
        self,
        *,
        sensor_id: int,
        device_id: int,
        base_severity: str,
        current_time: Optional[datetime] = None,
    ) -> OperationalContext:
        """Construye el contexto operacional completo."""
        
        now = current_time or datetime.now(timezone.utc)
        
        # Determinar turno y disponibilidad
        shift = self._shift_calc.determine_shift(now)
        is_weekend = self._shift_calc.is_weekend(now)
        is_holiday = self._shift_calc.is_holiday(now)
        
        if is_weekend or is_holiday:
            shift = WorkShift.WEEKEND
        
        staff = self._shift_calc.get_staff_availability(shift)
        response_time = self._shift_calc.get_response_time(staff)
        
        # Determinar impacto en producción
        production_impact, affected = self._impact_assessor.assess_impact(device_id, base_severity)
        
        # Obtener historial de mantenimiento
        maintenance = self._impact_assessor.get_maintenance_history(device_id)
        
        # Obtener incidentes recientes
        incidents = self._impact_assessor.get_recent_incidents(sensor_id)
        
        # Calcular multiplicador de severidad
        multiplier = self._severity_calc.calculate_multiplier(
            shift=shift,
            staff=staff,
            production_impact=production_impact,
            incidents=incidents,
            days_since_maintenance=maintenance.get("days_since_last"),
        )
        
        # Determinar si se debe aumentar urgencia
        urgency_boost = self._severity_calc.should_boost_urgency(
            base_severity=base_severity,
            shift=shift,
            incidents=incidents,
            production_impact=production_impact,
        )
        
        return OperationalContext(
            current_time=now,
            work_shift=shift,
            is_business_hours=self._shift_calc.is_business_hours(now),
            is_weekend=is_weekend,
            is_holiday=is_holiday,
            staff_availability=staff,
            response_time_minutes=response_time,
            production_impact=production_impact,
            affected_processes=affected,
            days_since_last_maintenance=maintenance.get("days_since_last"),
            pending_maintenance_tasks=maintenance.get("pending_tasks", 0),
            recent_incidents_count=incidents,
            severity_multiplier=multiplier,
            urgency_boost=urgency_boost,
        )


def adjust_severity_with_context(
    conn: Connection,
    *,
    sensor_id: int,
    device_id: int,
    base_severity: str,
) -> tuple[str, OperationalContext]:
    """Ajusta la severidad considerando el contexto operacional.
    
    Esta función es el punto de entrada principal para el batch runner.
    
    Returns:
        (adjusted_severity, operational_context)
    """
    builder = OperationalContextBuilder(conn)
    context = builder.build(
        sensor_id=sensor_id,
        device_id=device_id,
        base_severity=base_severity,
    )
    
    severity_calc = SeverityCalculator()
    adjusted_severity = severity_calc.adjust_severity(
        base_severity=base_severity,
        multiplier=context.severity_multiplier,
        urgency_boost=context.urgency_boost,
    )
    
    logger.debug(
        "[OPERATIONAL_CONTEXT] sensor_id=%s base=%s adjusted=%s multiplier=%.2f boost=%s",
        sensor_id, base_severity, adjusted_severity, context.severity_multiplier, context.urgency_boost
    )
    
    return adjusted_severity, context
