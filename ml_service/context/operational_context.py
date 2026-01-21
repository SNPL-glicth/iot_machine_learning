"""Módulo de Contexto Operacional para ML.

FALENCIA 4: La severidad se calcula con reglas estáticas sin considerar contexto operacional.

Este módulo implementa:
- Ajuste de severidad según hora del día (turno de trabajo)
- Consideración del historial de mantenimiento
- Evaluación del impacto en producción
- Disponibilidad de personal para respuesta
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


class WorkShift(Enum):
    """Turnos de trabajo."""
    MORNING = "morning"      # 06:00 - 14:00
    AFTERNOON = "afternoon"  # 14:00 - 22:00
    NIGHT = "night"          # 22:00 - 06:00
    WEEKEND = "weekend"      # Sábado y Domingo


class StaffAvailability(Enum):
    """Disponibilidad de personal."""
    FULL = "full"            # Personal completo disponible
    REDUCED = "reduced"      # Personal reducido
    MINIMAL = "minimal"      # Personal mínimo (guardias)
    NONE = "none"            # Sin personal en sitio


class ProductionImpact(Enum):
    """Impacto en producción."""
    CRITICAL = "critical"    # Detiene producción
    HIGH = "high"            # Afecta significativamente
    MEDIUM = "medium"        # Afecta parcialmente
    LOW = "low"              # Impacto mínimo
    NONE = "none"            # Sin impacto


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


class OperationalContextBuilder:
    """Builder para construir OperationalContext."""
    
    # Configuración de turnos (puede externalizarse a BD o config)
    SHIFT_CONFIG = {
        WorkShift.MORNING: (time(6, 0), time(14, 0)),
        WorkShift.AFTERNOON: (time(14, 0), time(22, 0)),
        WorkShift.NIGHT: (time(22, 0), time(6, 0)),
    }
    
    # Disponibilidad de personal por turno
    STAFF_BY_SHIFT = {
        WorkShift.MORNING: StaffAvailability.FULL,
        WorkShift.AFTERNOON: StaffAvailability.FULL,
        WorkShift.NIGHT: StaffAvailability.REDUCED,
        WorkShift.WEEKEND: StaffAvailability.MINIMAL,
    }
    
    # Tiempo de respuesta estimado por disponibilidad (minutos)
    RESPONSE_TIMES = {
        StaffAvailability.FULL: 15,
        StaffAvailability.REDUCED: 30,
        StaffAvailability.MINIMAL: 60,
        StaffAvailability.NONE: 120,
    }
    
    def __init__(self, conn: Connection):
        self._conn = conn
    
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
        shift = self._determine_shift(now)
        is_weekend = now.weekday() >= 5
        is_holiday = self._is_holiday(now)
        
        if is_weekend or is_holiday:
            shift = WorkShift.WEEKEND
        
        staff = self.STAFF_BY_SHIFT.get(shift, StaffAvailability.REDUCED)
        response_time = self.RESPONSE_TIMES.get(staff, 30)
        
        # Determinar impacto en producción
        production_impact, affected = self._assess_production_impact(device_id, base_severity)
        
        # Obtener historial de mantenimiento
        maintenance = self._get_maintenance_history(device_id)
        
        # Obtener incidentes recientes
        incidents = self._get_recent_incidents(sensor_id)
        
        # Calcular multiplicador de severidad
        multiplier = self._calculate_severity_multiplier(
            shift=shift,
            staff=staff,
            production_impact=production_impact,
            incidents=incidents,
            days_since_maintenance=maintenance.get("days_since_last"),
        )
        
        # Determinar si se debe aumentar urgencia
        urgency_boost = self._should_boost_urgency(
            base_severity=base_severity,
            shift=shift,
            incidents=incidents,
            production_impact=production_impact,
        )
        
        return OperationalContext(
            current_time=now,
            work_shift=shift,
            is_business_hours=self._is_business_hours(now),
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
    
    def _determine_shift(self, dt: datetime) -> WorkShift:
        """Determina el turno de trabajo actual."""
        current_time = dt.time()
        
        # Turno de mañana: 06:00 - 14:00
        if time(6, 0) <= current_time < time(14, 0):
            return WorkShift.MORNING
        
        # Turno de tarde: 14:00 - 22:00
        if time(14, 0) <= current_time < time(22, 0):
            return WorkShift.AFTERNOON
        
        # Turno de noche: 22:00 - 06:00
        return WorkShift.NIGHT
    
    def _is_business_hours(self, dt: datetime) -> bool:
        """Verifica si es horario laboral normal."""
        if dt.weekday() >= 5:  # Fin de semana
            return False
        
        current_time = dt.time()
        return time(8, 0) <= current_time <= time(18, 0)
    
    def _is_holiday(self, dt: datetime) -> bool:
        """Verifica si es día festivo.
        
        TODO: Implementar consulta a tabla de festivos o API externa.
        Por ahora retorna False.
        """
        return False
    
    def _assess_production_impact(
        self,
        device_id: int,
        base_severity: str,
    ) -> tuple[ProductionImpact, list[str]]:
        """Evalúa el impacto en producción del dispositivo afectado."""
        
        affected_processes = []
        
        try:
            # Obtener tipo de dispositivo y metadata
            row = self._conn.execute(
                text("""
                    SELECT device_type, metadata, name
                    FROM dbo.devices
                    WHERE id = :device_id
                """),
                {"device_id": device_id},
            ).fetchone()
            
            if row:
                device_type = str(row.device_type or "").lower()
                device_name = str(row.name or "")
                
                # Heurística de impacto por tipo de dispositivo
                critical_types = {"server", "datacenter", "production", "critical"}
                high_types = {"hvac", "power", "ups", "network"}
                medium_types = {"sensor_hub", "gateway", "monitoring"}
                
                if any(t in device_type for t in critical_types):
                    affected_processes.append(f"Producción ({device_name})")
                    return ProductionImpact.CRITICAL, affected_processes
                
                if any(t in device_type for t in high_types):
                    affected_processes.append(f"Infraestructura ({device_name})")
                    return ProductionImpact.HIGH, affected_processes
                
                if any(t in device_type for t in medium_types):
                    affected_processes.append(f"Monitoreo ({device_name})")
                    return ProductionImpact.MEDIUM, affected_processes
        
        except Exception:
            pass
        
        # Ajustar por severidad base
        if base_severity.lower() == "critical":
            return ProductionImpact.HIGH, affected_processes
        
        return ProductionImpact.LOW, affected_processes
    
    def _get_maintenance_history(self, device_id: int) -> dict:
        """Obtiene historial de mantenimiento del dispositivo.
        
        TODO: Implementar consulta a tabla de mantenimientos cuando exista.
        Por ahora usa heurísticas basadas en eventos.
        """
        
        try:
            # Usar eventos resueltos como proxy de mantenimiento
            row = self._conn.execute(
                text("""
                    SELECT 
                        DATEDIFF(day, MAX(resolved_at), GETDATE()) AS days_since_last,
                        COUNT(CASE WHEN status = 'active' THEN 1 END) AS pending
                    FROM dbo.ml_events
                    WHERE device_id = :device_id
                      AND resolved_at IS NOT NULL
                """),
                {"device_id": device_id},
            ).fetchone()
            
            if row:
                return {
                    "days_since_last": int(row.days_since_last) if row.days_since_last else None,
                    "pending_tasks": int(row.pending) if row.pending else 0,
                }
        except Exception:
            pass
        
        return {"days_since_last": None, "pending_tasks": 0}
    
    def _get_recent_incidents(self, sensor_id: int, days: int = 7) -> int:
        """Obtiene cantidad de incidentes recientes para el sensor."""
        
        try:
            row = self._conn.execute(
                text("""
                    SELECT COUNT(*) AS cnt
                    FROM dbo.ml_events
                    WHERE sensor_id = :sensor_id
                      AND created_at >= DATEADD(day, -:days, GETDATE())
                """),
                {"sensor_id": sensor_id, "days": days},
            ).fetchone()
            
            if row:
                return int(row.cnt)
        except Exception:
            pass
        
        return 0
    
    def _calculate_severity_multiplier(
        self,
        *,
        shift: WorkShift,
        staff: StaffAvailability,
        production_impact: ProductionImpact,
        incidents: int,
        days_since_maintenance: Optional[int],
    ) -> float:
        """Calcula el multiplicador de severidad basado en contexto operacional.
        
        Rango: 0.5 (reducir severidad) a 2.0 (aumentar severidad)
        Base: 1.0 (sin cambio)
        """
        
        multiplier = 1.0
        
        # Ajuste por turno (menos personal = más crítico)
        if shift == WorkShift.NIGHT:
            multiplier += 0.2
        elif shift == WorkShift.WEEKEND:
            multiplier += 0.3
        
        # Ajuste por disponibilidad de personal
        if staff == StaffAvailability.MINIMAL:
            multiplier += 0.2
        elif staff == StaffAvailability.NONE:
            multiplier += 0.4
        
        # Ajuste por impacto en producción
        if production_impact == ProductionImpact.CRITICAL:
            multiplier += 0.3
        elif production_impact == ProductionImpact.HIGH:
            multiplier += 0.2
        
        # Ajuste por incidentes recurrentes
        if incidents >= 5:
            multiplier += 0.3
        elif incidents >= 3:
            multiplier += 0.15
        
        # Ajuste por mantenimiento atrasado
        if days_since_maintenance is not None:
            if days_since_maintenance > 90:
                multiplier += 0.2
            elif days_since_maintenance > 60:
                multiplier += 0.1
        
        # Limitar rango
        return max(0.5, min(2.0, multiplier))
    
    def _should_boost_urgency(
        self,
        *,
        base_severity: str,
        shift: WorkShift,
        incidents: int,
        production_impact: ProductionImpact,
    ) -> bool:
        """Determina si se debe aumentar la urgencia de la alerta."""
        
        # Siempre boost si es crítico en turno nocturno/fin de semana
        if base_severity.lower() == "critical" and shift in (WorkShift.NIGHT, WorkShift.WEEKEND):
            return True
        
        # Boost si hay muchos incidentes recurrentes
        if incidents >= 5:
            return True
        
        # Boost si afecta producción crítica
        if production_impact == ProductionImpact.CRITICAL:
            return True
        
        return False


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
    
    # Mapear severidad base a valor numérico
    severity_values = {
        "info": 1,
        "warning": 2,
        "critical": 3,
    }
    
    base_value = severity_values.get(base_severity.lower(), 1)
    
    # Aplicar multiplicador
    adjusted_value = base_value * context.severity_multiplier
    
    # Aplicar boost de urgencia
    if context.urgency_boost and adjusted_value < 3:
        adjusted_value += 0.5
    
    # Mapear de vuelta a severidad
    if adjusted_value >= 2.5:
        adjusted_severity = "critical"
    elif adjusted_value >= 1.5:
        adjusted_severity = "warning"
    else:
        adjusted_severity = "info"
    
    logger.debug(
        "[OPERATIONAL_CONTEXT] sensor_id=%s base=%s adjusted=%s multiplier=%.2f boost=%s",
        sensor_id, base_severity, adjusted_severity, context.severity_multiplier, context.urgency_boost
    )
    
    return adjusted_severity, context
