"""Severity calculator service for operational context."""

from __future__ import annotations

from typing import Optional

from ..models.work_shift import WorkShift, StaffAvailability, ProductionImpact


class SeverityCalculator:
    """Calculates severity multiplier and urgency boost."""
    
    def calculate_multiplier(
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
    
    def should_boost_urgency(
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
    
    def adjust_severity(
        self,
        base_severity: str,
        multiplier: float,
        urgency_boost: bool,
    ) -> str:
        """Ajusta la severidad basada en multiplicador y boost."""
        severity_values = {
            "info": 1,
            "warning": 2,
            "critical": 3,
        }
        
        base_value = severity_values.get(base_severity.lower(), 1)
        adjusted_value = base_value * multiplier
        
        if urgency_boost and adjusted_value < 3:
            adjusted_value += 0.5
        
        if adjusted_value >= 2.5:
            return "critical"
        elif adjusted_value >= 1.5:
            return "warning"
        else:
            return "info"
