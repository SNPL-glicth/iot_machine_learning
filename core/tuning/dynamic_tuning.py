"""Orquesta dynamic tuning usando convergence detection + bounds enforcement.

Resuelve ADP-3 y ADP-4 integrando los nuevos componentes.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from core.tuning.convergence_detector import ConvergenceDetector, ConvergenceStatus
from core.parameters.parameter_bounds import ParameterBoundsEnforcer

logger = logging.getLogger(__name__)


class DynamicTuner:
    """
    Tuning dinámico de parámetros adaptativos con convergencia y bounds.

    Integra:
    - ConvergenceDetector: detecta oscilación/divergencia
    - ParameterBoundsEnforcer: previene explosion/colapso
    - ParameterRegistry: audit trail de cambios (opcional)

    Uso:
        tuner = DynamicTuner(registry=registry)
        new_alpha = tuner.tune_learning_rate(
            name="ML_BAYES_ALPHA",
            current_value=0.15,
            performance_metric=0.82,
            series_id="sensor_001"
        )
    """

    def __init__(
        self,
        registry: Optional[object] = None,
        bounds_enforcer: Optional[ParameterBoundsEnforcer] = None,
        convergence_window: int = 20,
    ) -> None:
        self._registry = registry
        self._bounds_enforcer = bounds_enforcer or ParameterBoundsEnforcer()
        self._convergence_window = convergence_window

        # Per-parameter convergence detectors
        self._detectors: Dict[str, ConvergenceDetector] = {}

    def _get_detector(self, name: str) -> ConvergenceDetector:
        """Obtiene o crea detector para un parámetro."""
        if name not in self._detectors:
            self._detectors[name] = ConvergenceDetector(window=self._convergence_window)
        return self._detectors[name]

    def tune_learning_rate(
        self,
        name: str,
        current_value: float,
        performance_metric: float,
        series_id: Optional[str] = None,
    ) -> float:
        """
        Ajusta learning rate basado en performance.

        - Si OSCILLATING: reduce alpha 20%
        - Si DIVERGING: reduce alpha 40%, log ERROR
        - Si CONVERGED: mantener (no tocar)
        - Si CONVERGING: ajuste normal
        Aplica bounds enforcement siempre.
        """
        detector = self._get_detector(name)
        result = detector.update(current_value)

        new_value = current_value

        if result.status == ConvergenceStatus.CONVERGED:
            # No change, already stable
            logger.info(
                "learning_rate_converged",
                extra={
                    "parameter": name,
                    "value": round(current_value, 6),
                    "series_id": series_id,
                },
            )
        elif result.status == ConvergenceStatus.OSCILLATING:
            # Reduce by 20%
            new_value = current_value * 0.8
            logger.warning(
                "learning_rate_oscillating_reduced",
                extra={
                    "parameter": name,
                    "old_value": round(current_value, 6),
                    "new_value": round(new_value, 6),
                    "oscillation_count": result.oscillation_count,
                    "series_id": series_id,
                },
            )
        elif result.status == ConvergenceStatus.DIVERGING:
            # Reduce by 40%
            new_value = current_value * 0.6
            logger.error(
                "learning_rate_diverging_reduced",
                extra={
                    "parameter": name,
                    "old_value": round(current_value, 6),
                    "new_value": round(new_value, 6),
                    "series_id": series_id,
                },
            )
        else:
            # CONVERGING or INSUFFICIENT_DATA: normal adjustment
            # Simple gradient: increase if performance good, decrease if bad
            if performance_metric > 0.7:
                new_value = current_value * 1.05
            elif performance_metric < 0.5:
                new_value = current_value * 0.95

        # Enforce bounds
        bounds_result = self._bounds_enforcer.enforce(name, new_value)

        # Update registry if available
        if self._registry is not None and hasattr(self._registry, "set_value"):
            try:
                self._registry.set_value(
                    name,
                    bounds_result.clipped_value,
                    f"dynamic_tuning: {result.status.value}, perf={performance_metric:.3f}",
                    "dynamic_tuner",
                )
            except Exception as e:
                logger.warning(
                    "registry_update_failed",
                    extra={"parameter": name, "error": str(e)},
                )

        return bounds_result.clipped_value

    def tune_contamination(
        self,
        current_value: float,
        false_positive_rate: float,
        target_fp_rate: float = 0.01,
        series_id: Optional[str] = None,
    ) -> float:
        """
        Ajusta contamination basado en FP rate observado.

        - Si FP > target * 1.5: reducir contamination 10%
        - Si FP < target * 0.5: aumentar contamination 5%
        Aplica bounds enforcement siempre.
        """
        name = "contamination"
        new_value = current_value

        if false_positive_rate > target_fp_rate * 1.5:
            # Too many false positives, reduce contamination
            new_value = current_value * 0.9
            logger.info(
                "contamination_reduced_high_fp",
                extra={
                    "old_value": round(current_value, 6),
                    "new_value": round(new_value, 6),
                    "fp_rate": round(false_positive_rate, 4),
                    "target": target_fp_rate,
                    "series_id": series_id,
                },
            )
        elif false_positive_rate < target_fp_rate * 0.5:
            # Too few false positives, can increase contamination
            new_value = current_value * 1.05
            logger.info(
                "contamination_increased_low_fp",
                extra={
                    "old_value": round(current_value, 6),
                    "new_value": round(new_value, 6),
                    "fp_rate": round(false_positive_rate, 4),
                    "target": target_fp_rate,
                    "series_id": series_id,
                },
            )

        # Enforce bounds
        bounds_result = self._bounds_enforcer.enforce(name, new_value)

        return bounds_result.clipped_value

    def get_convergence_report(self) -> Dict:
        """Reporte de estado de convergencia de todos los parámetros."""
        report = {}
        for name, detector in self._detectors.items():
            history = detector.get_history()
            if history:
                last_result = detector.update(history[-1])
                report[name] = {
                    "status": last_result.status.value,
                    "current_value": last_result.current_value,
                    "delta_mean": round(last_result.delta_mean, 6),
                    "oscillation_count": last_result.oscillation_count,
                    "steps_since_change": last_result.steps_since_change,
                    "recommendation": last_result.recommendation,
                }
        return report
