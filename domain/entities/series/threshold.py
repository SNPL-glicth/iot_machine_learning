"""Entidad Threshold — definición de umbral de alerta.

Value object puro del dominio.  Usado por SeverityRules y ThresholdPolicy
para determinar si un valor predicho viola los límites configurados.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Threshold:
    """Umbral de alerta con reglas de severidad.

    Attributes:
        value_min: Límite inferior (puede ser None).
        value_max: Límite superior (puede ser None).
        condition_type: Tipo de condición, e.g. "greater_than", "less_than".
        severity: Severidad asignada cuando se viola el umbral.
    """

    value_min: Optional[float] = None
    value_max: Optional[float] = None
    condition_type: str = "greater_than"
    severity: str = "critical"

    def severity_for(self, value: float) -> str:
        """Devuelve la severidad si *value* viola el umbral, sino 'none'."""
        if self._is_violated(value):
            return self.severity
        return "none"

    def risk_level_for(self, value: float) -> str:
        """Devuelve el nivel de riesgo para *value*."""
        if not self._is_violated(value):
            return "normal"
        # Mapeo simple severidad → risk_level
        mapping = {
            "critical": "critical",
            "high": "high",
            "warning": "medium",
            "medium": "medium",
            "low": "low",
        }
        return mapping.get(self.severity, "high")

    def _is_violated(self, value: float) -> bool:
        """Evalúa si *value* incumple el umbral."""
        cond = self.condition_type
        vmin = self.value_min
        vmax = self.value_max

        if cond == "greater_than" and vmin is not None:
            return value > vmin
        if cond == "less_than" and vmin is not None:
            return value < vmin
        if cond == "between" and vmin is not None and vmax is not None:
            return not (vmin <= value <= vmax)
        if cond == "outside" and vmin is not None and vmax is not None:
            return value < vmin or value > vmax
        return False
