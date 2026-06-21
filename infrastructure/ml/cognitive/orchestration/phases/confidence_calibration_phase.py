"""Confidence Calibration Phase — Temperature-scaled sigmoid calibration.

Single responsibility: convert raw confidence scores into calibrated probabilities
using temperature scaling with regime-aware adjustment.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import PipelineContext

from iot_machine_learning.infrastructure.ml.calibration import ConfidenceCalibrator

logger = logging.getLogger(__name__)


class ConfidenceCalibrationPhase:
    """Phase: Confidence calibration using temperature scaling.
    
    Converts raw confidence scores [0, +inf) into calibrated probabilities [0, 1]
    using sigmoid transformation with configurable temperature.
    
    Supports regime-aware temperature adjustment for adaptive calibration.
    """
    
    def __init__(
        self,
        calibrator: Optional[ConfidenceCalibrator] = None,
    ) -> None:
        """Initialize confidence calibration phase.
        
        Args:
            calibrator: Optional calibrator instance. If None, created from config.
        """
        self._calibrator = calibrator
    
    @property
    def name(self) -> str:
        return "confidence_calibration"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute confidence calibration if enabled.
        
        Args:
            ctx: Pipeline context with raw confidence.
        
        Returns:
            Updated context with calibrated confidence.
        """
        # Check if calibration is enabled
        enabled = ctx.flags.get("ML_ENABLE_CONFIDENCE_CALIBRATION", True)
        
        if not enabled:
            logger.debug(
                "confidence_calibration_disabled",
                extra={
                    "series_id": ctx.series_id,
                    "event": "PHASE_SKIPPED",
                },
            )
            return ctx
        
        try:
            # Lazy-initialize calibrator from config
            if self._calibrator is None:
                self._calibrator = self._create_calibrator_from_config(ctx.flags)
            
            # Get raw confidence
            raw_confidence = ctx.fused_confidence
            
            if raw_confidence is None:
                logger.debug(
                    "confidence_calibration_no_raw_confidence",
                    extra={
                        "series_id": ctx.series_id,
                        "event": "PHASE_SKIPPED",
                    },
                )
                return ctx
            
            # Extract regime from metadata
            regime = self._extract_regime(ctx)
            
            # Calibrate confidence
            calibrated_confidence = self._calibrator.calibrate(
                score=raw_confidence,
                regime=regime,
            )
            
            logger.info(
                "confidence_calibration_applied",
                extra={
                    "series_id": ctx.series_id,
                    "event": "CONFIDENCE_CALIBRATED",
                    "raw_confidence": round(raw_confidence, 4),
                    "calibrated_confidence": round(calibrated_confidence, 4),
                    "regime": regime,
                },
            )
            
            # Update context with calibrated confidence AND preserve raw
            result_ctx = ctx.with_field(
                raw_fused_confidence=raw_confidence,
                fused_confidence=calibrated_confidence
            )
            
            # Record confidence calibration metrics (Phase 3C)
            if ctx.metrics_collector is not None:
                try:
                    ctx.metrics_collector.record_confidence_calibration(
                        raw_confidence=raw_confidence,
                        calibrated_confidence=calibrated_confidence,
                    )
                except Exception as e:
                    logger.debug(f"metrics_collection_failed: {e}")
            
            return result_ctx
        
        except Exception as e:
            logger.error(
                "confidence_calibration_failed",
                extra={
                    "series_id": ctx.series_id,
                    "event": "PHASE_ERROR",
                    "error": str(e),
                    "action_taken": "skip_calibration",
                },
            )
            return ctx
    
    def _create_calibrator_from_config(self, flags: dict) -> ConfidenceCalibrator:
        """Create calibrator from config flags.
        
        Args:
            flags: Configuration flags.
        
        Returns:
            ConfidenceCalibrator instance.
        """
        temperature = flags.get("ML_CONFIDENCE_TEMPERATURE", 1.5)
        
        # Build regime temperatures from config
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
    
    def _extract_regime(self, ctx: PipelineContext) -> Optional[str]:
        """Extract regime from context metadata.
        
        Args:
            ctx: Pipeline context.
        
        Returns:
            Regime name or None.
        """
        try:
            # Try to get regime from cognitive diagnostic
            if hasattr(ctx, 'metadata') and ctx.metadata:
                cognitive_diag = ctx.metadata.get('cognitive_diagnostic', {})
                if cognitive_diag:
                    return cognitive_diag.get('regime')
            
            # Fallback: try to get from profile
            if hasattr(ctx, 'profile') and ctx.profile:
                return getattr(ctx.profile, 'regime', None)
            
            return None
        
        except Exception as e:
            logger.debug(
                "regime_extraction_failed",
                extra={
                    "error": str(e),
                    "action_taken": "use_default_temperature",
                },
            )
            return None
