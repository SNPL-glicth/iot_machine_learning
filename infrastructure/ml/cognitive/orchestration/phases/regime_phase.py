"""Regime Phase — Regime detection pipeline integration.

Integrates RegimeDetectionPipeline for advanced regime detection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import PipelineContext

try:
    from ...regime import RegimeDetectionPipeline, OperationalRegimeClassifier
except (ImportError, ModuleNotFoundError):
    RegimeDetectionPipeline = None  # type: ignore[assignment,misc]
    OperationalRegimeClassifier = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class RegimePhase:
    """Phase: Regime detection.
    
    Uses RegimeDetectionPipeline for advanced operational regime
    detection with context-aware classification.
    """
    
    def __init__(self, regime_detection_pipeline: Optional[Any] = None) -> None:
        """Initialize regime phase.
        
        Args:
            regime_detection_pipeline: Optional RegimeDetectionPipeline instance.
        """
        self._regime_detection_pipeline = regime_detection_pipeline
    
    @property
    def name(self) -> str:
        return "regime"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute regime phase.
        
        Args:
            ctx: Pipeline context with cognitive metrics.
        
        Returns:
            Updated context with regime detection results.
        """
        # Skip if no regime detection pipeline available
        if self._regime_detection_pipeline is None or ctx.regime_detection_pipeline is None:
            return ctx
        
        try:
            # Use regime detection pipeline from context if available
            regime_pipeline = ctx.regime_detection_pipeline if ctx.regime_detection_pipeline else self._regime_detection_pipeline
            
            # Prepare data for regime detection
            if ctx.values and ctx.timestamps and len(ctx.values) > 20:
                try:
                    # Detect regime
                    regime_result = regime_pipeline.detect_regime(
                        sensor_id=ctx.series_id,
                        values=ctx.values,
                        timestamps=ctx.timestamps,
                    )
                    
                    # Extract regime detection results
                    detected_regime = getattr(regime_result, 'regime', ctx.regime or 'unknown')
                    regime_confidence = getattr(regime_result, 'confidence', 0.0)
                    regime_transition = getattr(regime_result, 'is_transition', False)
                    regime_stability = getattr(regime_result, 'stability', 'stable')
                    
                    # Log regime detection summary
                    logger.debug(
                        "regime_detection_completed",
                        extra={
                            "series_id": ctx.series_id,
                            "detected_regime": detected_regime,
                            "regime_confidence": regime_confidence,
                            "regime_transition": regime_transition,
                        },
                    )
                    
                    return ctx.with_field(
                        regime_result=regime_result,
                        detected_regime=detected_regime,
                        regime_confidence=regime_confidence,
                        regime_transition=regime_transition,
                        regime_stability=regime_stability,
                    )
                except Exception as e:
                    logger.debug(f"regime_detection_failed: {e}")
            else:
                logger.debug(
                    "regime_phase_skipped_insufficient_data",
                    extra={"series_id": ctx.series_id, "n_values": len(ctx.values) if ctx.values else 0},
                )
        
        except Exception as e:
            logger.debug(f"regime_phase_failed: {e}")
        
        return ctx
