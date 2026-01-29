"""Work shift and availability enums for operational context."""

from enum import Enum


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
