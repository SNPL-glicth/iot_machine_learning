"""Shift calculator service for operational context."""

from __future__ import annotations

from datetime import datetime, time

from ..models.work_shift import WorkShift, StaffAvailability


class ShiftCalculator:
    """Calculates work shift and staff availability."""
    
    # Configuración de turnos
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
    
    def determine_shift(self, dt: datetime) -> WorkShift:
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
    
    def is_business_hours(self, dt: datetime) -> bool:
        """Verifica si es horario laboral normal."""
        if dt.weekday() >= 5:  # Fin de semana
            return False
        
        current_time = dt.time()
        return time(8, 0) <= current_time <= time(18, 0)
    
    def is_weekend(self, dt: datetime) -> bool:
        """Verifica si es fin de semana."""
        return dt.weekday() >= 5
    
    def is_holiday(self, dt: datetime) -> bool:
        """Verifica si es día festivo.
        
        TODO: Implementar consulta a tabla de festivos o API externa.
        """
        return False
    
    def get_staff_availability(self, shift: WorkShift) -> StaffAvailability:
        """Obtiene disponibilidad de personal para un turno."""
        return self.STAFF_BY_SHIFT.get(shift, StaffAvailability.REDUCED)
    
    def get_response_time(self, availability: StaffAvailability) -> int:
        """Obtiene tiempo de respuesta estimado en minutos."""
        return self.RESPONSE_TIMES.get(availability, 30)
