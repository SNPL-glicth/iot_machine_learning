"""Temperature scaling matemáticamente justificado para confidence calibration.

Resuelve CONF-1: temperatura sin fórmula documentada.

Fórmula implementada (Platt scaling generalizado):
    scaled = sigmoid((confidence - 0.5) / T)
    donde sigmoid(x) = 1 / (1 + exp(-x))

T > 1: aplana distribución (más incertidumbre)
T < 1: agudiza distribución (más certeza)
T = 1: identidad aproximada alrededor de 0.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.parameters.numerical_constants import CONFIDENCE


@dataclass
class TemperatureScalingResult:
    """Resultado del temperature scaling."""
    original_confidence: float
    scaled_confidence: float
    temperature_used: float
    regime: str
    formula: str


class TemperatureScaler:
    """
    Temperature Scaling para confidence calibration.

    FÓRMULA: sigmoid((confidence - 0.5) / T)
    - Centra en 0.5 antes de escalar
    - Input asumido en [0, 1]
    - T > 1: aplana distribución (más incertidumbre)
    - T < 1: agudiza distribución (más certeza)

    CUÁNDO USAR ESTE MÉTODO:
    - Post-fusión: calibración probabilística DESPUÉS de fusionar engines
    - Regime-aware: cuando se conoce el régimen (STABLE, VOLATILE, etc.)
    - Confidence values: para confidence en [0, 1], NO para anomaly scores
    - NO usar para: anomaly scores (usar infrastructure/ml/calibration/confidence_calibrator.py)
    - NO usar para: pre-decisión por calidad de datos (usar domain/services/confidence_calibrator.py)

    DIFERENCIA con domain/services/confidence_calibrator.py:
    - Este módulo: sigmoid centrado en 0.5, input en [0,1]
    - domain service: penalidades aditivas sobre raw_confidence
    - Son métodos COMPLEMENTARIOS, no duplicados:
      * TemperatureScaler: calibración probabilística post-fusión
      * PenaltyCalibrator: ajuste por calidad de datos pre-decisión

    FÓRMULA DE TEMPERATURA (DIFERENCIA INTENCIONAL):
    - Este módulo: sigmoid((c - 0.5) / T) donde c ∈ [0, 1]
      * Centra en 0.5 porque confidence tiene punto medio natural
      * c=0.5 → calibrated=0.5, c=1.0 → calibrated>0.5
    - InfraCalibrator: sigmoid(score / T) donde score ∈ [0, +inf)
      * NO centra porque anomaly scores no tienen punto medio natural
      * score=0 → calibrated≈0.5, score→∞ → calibrated→1.0
    """

    REGIME_TEMPERATURES: Dict[str, float] = {
        "STABLE": CONFIDENCE.TEMP_STABLE,
        "TRENDING": CONFIDENCE.TEMP_TRENDING,
        "VOLATILE": CONFIDENCE.TEMP_VOLATILE,
        "NOISY": CONFIDENCE.TEMP_NOISY,
        "DEFAULT": CONFIDENCE.TEMP_DEFAULT,
    }

    def __init__(
        self,
        default_temperature: float = CONFIDENCE.TEMP_DEFAULT,
        floor: float = CONFIDENCE.MIN_CONFIDENCE,
        ceiling: float = CONFIDENCE.MAX_CONFIDENCE,
    ) -> None:
        self._default_temperature = default_temperature
        self._floor = floor
        self._ceiling = ceiling

    def scale(
        self,
        confidence: float,
        regime: str = "DEFAULT",
        temperature_override: Optional[float] = None,
    ) -> TemperatureScalingResult:
        """Aplica temperature scaling con fórmula documentada."""
        # Determine temperature
        if temperature_override is not None:
            temperature = temperature_override
        else:
            temperature = self.REGIME_TEMPERATURES.get(regime, self._default_temperature)

        # Apply sigmoid scaling: sigmoid((confidence - 0.5) / T)
        x = (confidence - 0.5) / temperature
        # Overflow protection: exp(-x) overflows when x < -700
        x = max(-700.0, min(700.0, x))
        scaled = 1.0 / (1.0 + math.exp(-x))

        # Apply floor and ceiling
        scaled = max(self._floor, min(self._ceiling, scaled))

        formula = f"sigmoid((c - 0.5) / {temperature:.2f}) where c={confidence:.3f}"

        return TemperatureScalingResult(
            original_confidence=confidence,
            scaled_confidence=scaled,
            temperature_used=temperature,
            regime=regime,
            formula=formula,
        )

    def calibrate_temperature(
        self,
        confidences: List[float],
        target_mean: float = 0.5,
    ) -> float:
        """
        Calibra temperatura óptima para que mean(scaled) ≈ target_mean.
        Búsqueda por bisección en [0.5, 3.0].
        """
        if not confidences:
            return self._default_temperature

        def compute_mean_scaled(temp: float) -> float:
            scaled_values = []
            for conf in confidences:
                x = (conf - 0.5) / temp
                # Overflow protection: exp(-x) overflows when x < -700
                x = max(-700.0, min(700.0, x))
                scaled = 1.0 / (1.0 + math.exp(-x))
                scaled_values.append(scaled)
            return sum(scaled_values) / len(scaled_values)

        # Bisection search
        low, high = 0.5, 3.0
        tolerance = 0.01
        max_iterations = 50

        for _ in range(max_iterations):
            mid = (low + high) / 2.0
            mean_scaled = compute_mean_scaled(mid)

            if abs(mean_scaled - target_mean) < tolerance:
                return mid

            if mean_scaled < target_mean:
                # Need to increase scaled values → decrease temperature
                high = mid
            else:
                # Need to decrease scaled values → increase temperature
                low = mid

        return (low + high) / 2.0
