"""Confidence Calibration Phase — MED-1 Refactoring.

Confidence calibration when enabled.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import PipelineContext

from ......domain.services.confidence_calibrator import ConfidenceCalibrator

logger = logging.getLogger(__name__)


class ConfidenceCalibrationPhase:
    """Phase 8: Confidence calibration (optional)."""
    
    @property
    def name(self) -> str:
        return "confidence_calibration"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute confidence calibration if enabled."""
        flags = ctx.flags
        
        if not flags.ML_CONFIDENCE_CALIBRATION_ENABLED:
            return ctx
        
        try:
            calibrator = ConfidenceCalibrator()
            
            # Compute engine disagreement
            engine_disagreement = calibrator.compute_engine_disagreement(ctx.perceptions)
            
            # Determine baseline status
            only_baseline = (
                len(ctx.perceptions) == 1 and 
                ctx.perceptions[0].engine_name == "baseline"
            )
            all_inhibited = (
                all(s.inhibited_weight < 0.05 for s in ctx.inhibition_states) 
                if ctx.inhibition_states else False
            )
            
            # Get noise ratio
            noise_ratio = getattr(ctx.profile, 'noise_ratio', 0.0)
            
            # Check coherence conflict
            coherence_conflict = (
                ctx.coherence_result is not None and 
                not ctx.coherence_result.is_coherent
            )
            
            calibrated_confidence = calibrator.calibrate(
                raw_confidence=ctx.fused_confidence,
                n_points=len(ctx.values),
                noise_ratio=noise_ratio,
                engine_disagreement=engine_disagreement,
                only_baseline_active=only_baseline,
                coherence_conflict=coherence_conflict,
                all_engines_inhibited=all_inhibited,
            )
            
            logger.info("confidence_calibrated", extra={
                "series_id": ctx.series_id,
                "raw": round(calibrated_confidence.raw, 4),
                "calibrated": round(calibrated_confidence.calibrated, 4),
                "penalty": round(calibrated_confidence.penalty_applied, 4),
                "n_reasons": len(calibrated_confidence.reasons),
            })
            
            return ctx.with_field(
                calibrated_confidence=calibrated_confidence,
                fused_confidence=calibrated_confidence.calibrated,
            )
            
        except Exception as e:
            logger.debug(f"confidence_calibration_skipped: {e}")
            return ctx
