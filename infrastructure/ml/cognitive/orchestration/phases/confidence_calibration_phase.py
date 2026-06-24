"""Confidence Calibration Phase — unified temperature‑scaled sigmoid calibration.

Uses infrastructure/ml/calibration/ConfidenceCalibrator with:
  * floor=0.30, ceiling=0.95
  * data_quality_score boosts temperature when quality is low
  * regime‑aware base temperature
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import PipelineContext

from iot_machine_learning.infrastructure.ml.calibration import ConfidenceCalibrator

logger = logging.getLogger(__name__)


class ConfidenceCalibrationPhase:
    """Phase: confidence calibration using temperature‑scaled sigmoid."""

    def __init__(
        self,
        calibrator: Optional[ConfidenceCalibrator] = None,
    ) -> None:
        self._calibrator = calibrator

    @property
    def name(self) -> str:
        return "confidence_calibration"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        enabled = ctx.flags.get("ML_ENABLE_CONFIDENCE_CALIBRATION", True)
        if not enabled:
            return ctx

        try:
            if self._calibrator is None:
                self._calibrator = self._create_calibrator_from_config(ctx.flags)

            raw_confidence = getattr(ctx, "fused_confidence", None)
            if raw_confidence is None:
                return ctx

            regime = getattr(ctx, "regime", None)
            data_quality = getattr(ctx, "data_quality_score", 1.0)

            result = self._calibrator.calibrate(
                score=raw_confidence,
                regime=regime,
                data_quality=data_quality,
            )

            logger.info(
                "confidence_calibration_applied",
                extra={
                    "series_id": ctx.series_id,
                    "raw_confidence": round(raw_confidence, 4),
                    "calibrated_confidence": round(result.calibrated, 4),
                    "data_quality": round(data_quality, 4),
                    "regime": regime,
                    "reasons": result.reasons,
                },
            )

            result_ctx = ctx.with_field(
                raw_fused_confidence=raw_confidence,
                fused_confidence=result.calibrated,
            )

            if ctx.metrics_collector is not None:
                try:
                    ctx.metrics_collector.record_confidence_calibration(
                        raw_confidence=raw_confidence,
                        calibrated_confidence=result.calibrated,
                    )
                except Exception as e:
                    logger.debug(f"metrics_collection_failed: {e}")

            return result_ctx

        except Exception as e:
            logger.error(
                "confidence_calibration_failed",
                extra={
                    "series_id": ctx.series_id,
                    "error": str(e),
                },
            )
            return ctx

    def _create_calibrator_from_config(self, flags: dict) -> ConfidenceCalibrator:
        temperature = flags.get("ML_CONFIDENCE_TEMPERATURE", 1.5)
        regime_temperatures = {
            "VOLATILE": flags.get("ML_CONFIDENCE_TEMP_VOLATILE", 2.0),
            "STABLE": flags.get("ML_CONFIDENCE_TEMP_STABLE", 1.2),
            "TRENDING": flags.get("ML_CONFIDENCE_TEMP_TRENDING", 1.5),
            "NOISY": flags.get("ML_CONFIDENCE_TEMP_NOISY", 1.8),
        }
        return ConfidenceCalibrator(
            temperature=temperature,
            regime_temperatures=regime_temperatures,
        )
