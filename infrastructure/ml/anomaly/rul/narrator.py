"""RUL narrator — converts estimates to operator-friendly messages."""

from __future__ import annotations

from typing import Optional

from .models import RULEstimate


class RULNarrator:
    """Produce human-readable narratives from RUL estimates."""

    def narrate(self, estimate: Optional[RULEstimate]) -> str:
        """Return operator narrative or empty string if no estimate."""
        if estimate is None:
            return ""

        hours = estimate.time_to_failure_hours
        if hours is None:
            return ""

        if estimate.urgency == "CRITICAL":
            return (
                f"INTERVENCION URGENTE: ~{hours:.0f}h para falla estimada. "
                "Detencion programada recomendada inmediatamente."
            )

        if estimate.urgency == "MEDIUM":
            return (
                f"MONITOREO ACTIVO: ~{hours:.0f}h para falla estimada. "
                "Programar mantenimiento preventivo hoy."
            )

        return (
            f"ALERTA TEMPRANA: ~{hours:.0f}h para falla estimada. "
            "Incluir en proximo ciclo de mantenimiento."
        )
