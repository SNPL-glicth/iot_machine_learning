"""Validador de consistencia cross-module.

Principio: Open/Closed - extensible sin modificar.
Valida que los parámetros numéricos sean consistentes entre módulos.
USA ParameterRegistry como single source of truth.
"""

from __future__ import annotations

from typing import List, Optional

from core.parameters.parameter_registry import ParameterRegistry


class ParameterValidator:
    """Valida consistencia de parámetros usando registry."""

    def __init__(self, registry: Optional[ParameterRegistry] = None) -> None:
        self._registry = registry or ParameterRegistry()

    def validate_all(self) -> List[str]:
        """Retorna lista de inconsistencias detectadas."""
        issues = []

        # Epsilon hierarchy
        eps_comp = self._registry.get_value("EPSILON.COMPARISON")
        eps_div = self._registry.get_value("EPSILON.DIVISION")
        eps_conf = self._registry.get_value("EPSILON.CONFIDENCE")

        if eps_div >= eps_comp:
            issues.append(f"EPSILON.DIVISION ({eps_div}) debe ser < EPSILON.COMPARISON ({eps_comp})")
        if eps_div >= eps_conf:
            issues.append(f"EPSILON.DIVISION ({eps_div}) debe ser < EPSILON.CONFIDENCE ({eps_conf})")

        # Z-score monotonicity
        z_lower = self._registry.get_value("STAT_THRESHOLDS.Z_SCORE_LOWER")
        z_upper = self._registry.get_value("STAT_THRESHOLDS.Z_SCORE_UPPER")
        if z_upper <= z_lower:
            issues.append(f"Z_SCORE_UPPER ({z_upper}) debe ser > Z_SCORE_LOWER ({z_lower})")

        # Contamination range consistency
        cont_def = self._registry.get_value("STAT_THRESHOLDS.CONTAMINATION_DEFAULT")
        cont_min = self._registry.get_value("STAT_THRESHOLDS.CONTAMINATION_MIN")
        cont_max = self._registry.get_value("STAT_THRESHOLDS.CONTAMINATION_MAX")
        if not (cont_min < cont_def < cont_max):
            issues.append(f"CONTAMINATION_DEFAULT ({cont_def}) debe estar en ({cont_min}, {cont_max})")

        # Confidence range consistency
        conf_min = self._registry.get_value("CONFIDENCE.MIN_CONFIDENCE")
        conf_max = self._registry.get_value("CONFIDENCE.MAX_CONFIDENCE")
        if conf_min >= conf_max:
            issues.append(f"CONFIDENCE.MIN ({conf_min}) debe ser < CONFIDENCE.MAX ({conf_max})")

        # Statistical consistency: contamination vs 3σ
        expected_3sigma = 0.003
        if cont_def > expected_3sigma * 10:
            issues.append(f"CONTAMINATION ({cont_def}) es 10x mayor que tasa de 3σ ({expected_3sigma})")

        return issues

    def validate_epsilon_hierarchy(self) -> bool:
        """Valida jerarquía correcta de epsilons."""
        eps_comp = self._registry.get_value("EPSILON.COMPARISON")
        eps_div = self._registry.get_value("EPSILON.DIVISION")
        eps_conf = self._registry.get_value("EPSILON.CONFIDENCE")
        return eps_div < eps_comp and eps_div < eps_conf
