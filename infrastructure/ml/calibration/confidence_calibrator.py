"""Confidence calibrator — temperature scaling for confidence values [0, 1].

Single responsibility: convert raw confidence scores into calibrated probabilities
using temperature‑scaled sigmoid centred at 0.5.

Formula (Platt scaling generalised):
    calibrated = sigmoid((confidence - 0.5) / T)
    where sigmoid(x) = 1 / (1 + exp(-x))

T > 1 → flatter distribution (more uncertainty)
T < 1 → sharper distribution (more certainty)

Supports:
  * data_quality_score boosting of temperature (low quality → wider uncertainty)
  * floor=0.30, ceiling=0.95
  * regime‑aware base temperature
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.parameters.numerical_constants import CONFIDENCE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibratedConfidence:
    calibrated: float
    raw: float
    penalty_applied: float
    reasons: List[str] = field(default_factory=list)


class ConfidenceCalibrator:
    """Calibrates confidence [0, 1] using temperature‑scaled sigmoid.

    FLOOR / CEILING (unified):
      * floor=0.30  – matches CONFIDENCE.MIN_CONFIDENCE
      * ceiling=0.95 – matches CONFIDENCE.MAX_CONFIDENCE

    DATA QUALITY ADJUSTMENT:
      * data_quality < 0.5 → temperature × 1.3
      * data_quality < 0.3 → temperature × 1.6
    """

    FLOOR: float = 0.30
    CEILING: float = 0.95

    REGIME_TEMPERATURES: Dict[str, float] = {
        "STABLE": CONFIDENCE.TEMP_STABLE,
        "TRENDING": CONFIDENCE.TEMP_TRENDING,
        "VOLATILE": CONFIDENCE.TEMP_VOLATILE,
        "NOISY": CONFIDENCE.TEMP_NOISY,
        "DEFAULT": CONFIDENCE.TEMP_DEFAULT,
    }

    def __init__(
        self,
        temperature: float = CONFIDENCE.TEMP_DEFAULT,
        regime_temperatures: Optional[Dict[str, float]] = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self._base_temperature = temperature
        if regime_temperatures is not None:
            self._regime_temperatures = regime_temperatures

    def calibrate(
        self,
        score: float,
        regime: Optional[str] = None,
        data_quality: float = 1.0,
    ) -> CalibratedConfidence:
        """Calibrate confidence score.

        Args:
            score: Raw confidence [0, 1].
            regime: Optional regime name for temperature override.
            data_quality: Data quality score [0, 1] from SanitizePhase.

        Returns:
            CalibratedConfidence with audit trail.
        """
        reasons: List[str] = []

        if not math.isfinite(score):
            logger.warning("confidence_calibration_invalid_score", extra={"score": score})
            return CalibratedConfidence(
                calibrated=self.FLOOR,
                raw=score,
                penalty_applied=0.0,
                reasons=["invalid_score"],
            )

        temperature = self._get_temperature(regime)

        # Data quality adjustment
        if data_quality < 0.3:
            temperature *= 1.6
            reasons.append(f"data_quality={data_quality:.3f} < 0.3 → T×1.6")
        elif data_quality < 0.5:
            temperature *= 1.3
            reasons.append(f"data_quality={data_quality:.3f} < 0.5 → T×1.3")

        # sigmoid((c - 0.5) / T)
        x = (score - 0.5) / temperature
        x = max(-700.0, min(700.0, x))
        calibrated = 1.0 / (1.0 + math.exp(-x))

        raw = score
        calibrated = max(self.FLOOR, min(self.CEILING, calibrated))

        if calibrated < raw:
            reasons.append(f"temperature_scaling: T={temperature:.3f}")

        logger.debug(
            "confidence_calibration",
            extra={
                "raw": round(raw, 4),
                "calibrated": round(calibrated, 4),
                "temperature": round(temperature, 4),
                "regime": regime,
                "data_quality": round(data_quality, 4),
            },
        )

        return CalibratedConfidence(
            calibrated=calibrated,
            raw=raw,
            penalty_applied=raw - calibrated,
            reasons=reasons,
        )

    def _get_temperature(self, regime: Optional[str]) -> float:
        if regime and regime in self.REGIME_TEMPERATURES:
            return self.REGIME_TEMPERATURES[regime]
        return self._base_temperature
