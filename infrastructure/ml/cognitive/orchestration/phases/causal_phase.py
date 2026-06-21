"""Causal Phase — Operational causal mapping integration.

Integrates causal correlation, dependency graphs, and event propagation tracking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import PipelineContext

try:
    from ...causal import (
        CausalCorrelationEngine,
        EventPropagationTracker,
        PropagationConfidenceCalculator,
    )
except (ImportError, ModuleNotFoundError):
    CausalCorrelationEngine = None  # type: ignore[assignment,misc]
    EventPropagationTracker = None  # type: ignore[assignment,misc]
    PropagationConfidenceCalculator = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class CausalPhase:
    """Phase: Operational causal mapping.
    
    Detects causal correlations, tracks event propagation, and builds
    operational dependency graphs.
    """
    
    def __init__(
        self,
        causal_correlation_engine: Optional[Any] = None,
        event_propagation_tracker: Optional[Any] = None,
        propagation_confidence_calculator: Optional[Any] = None,
    ) -> None:
        """Initialize causal phase.
        
        Args:
            causal_correlation_engine: Optional CausalCorrelationEngine instance.
            event_propagation_tracker: Optional EventPropagationTracker instance.
            propagation_confidence_calculator: Optional PropagationConfidenceCalculator instance.
        """
        self._causal_correlation_engine = causal_correlation_engine
        if CausalCorrelationEngine is not None and self._causal_correlation_engine is None:
            self._causal_correlation_engine = CausalCorrelationEngine()
        
        self._event_propagation_tracker = event_propagation_tracker
        if EventPropagationTracker is not None and self._event_propagation_tracker is None:
            self._event_propagation_tracker = EventPropagationTracker()
        
        self._propagation_confidence_calculator = propagation_confidence_calculator
        if PropagationConfidenceCalculator is not None and self._propagation_confidence_calculator is None:
            self._propagation_confidence_calculator = PropagationConfidenceCalculator()
    
    @property
    def name(self) -> str:
        return "causal"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute causal phase.
        
        Args:
            ctx: Pipeline context with cognitive metrics.
        
        Returns:
            Updated context with causal metrics.
        """
        # Skip if no causal components available
        if self._causal_correlation_engine is None:
            return ctx
        
        try:
            # Detect causal correlations
            causal_correlations = []
            if self._causal_correlation_engine is not None and ctx.profile:
                try:
                    # Add current reading to correlation engine
                    if ctx.values:
                        self._causal_correlation_engine.add_reading(
                            sensor_id=int(ctx.series_id) if ctx.series_id.isdigit() else 0,
                            value=ctx.values[-1],
                            timestamp=ctx.timestamps[-1] if ctx.timestamps else None,
                        )
                    
                    # Detect correlations
                    correlations = self._causal_correlation_engine.detect_correlations(
                        sensor_id=int(ctx.series_id) if ctx.series_id.isdigit() else 0,
                    )
                    
                    causal_correlations = [
                        {
                            "target_sensor_id": corr.target_sensor_id,
                            "correlation_coefficient": corr.correlation_coefficient,
                            "lag_seconds": corr.lag_seconds,
                            "confidence": corr.confidence,
                        }
                        for corr in correlations
                    ]
                except Exception as e:
                    logger.debug(f"causal_correlation_detection_failed: {e}")
            
            # Track event propagation
            propagation_id = None
            if self._event_propagation_tracker is not None and ctx.profile:
                try:
                    z_score = getattr(ctx.profile, "z_score", 0.0)
                    if abs(z_score) > 2.5:  # Anomaly threshold
                        propagation_id = self._event_propagation_tracker.start_propagation(
                            source_sensor_id=int(ctx.series_id) if ctx.series_id.isdigit() else 0,
                            timestamp=ctx.timestamps[-1] if ctx.timestamps else None,
                        )
                except Exception as e:
                    logger.debug(f"event_propagation_tracking_failed: {e}")
            
            # Log causal summary
            logger.debug(
                "causal_phase_completed",
                extra={
                    "series_id": ctx.series_id,
                    "n_correlations": len(causal_correlations),
                    "propagation_started": propagation_id is not None,
                },
            )
            
            return ctx.with_field(
                causal_correlations=causal_correlations,
                propagation_id=propagation_id,
                causal_context={
                    "has_causal_analysis": True,
                    "n_correlations": len(causal_correlations),
                },
            )
        
        except Exception as e:
            logger.debug(f"causal_phase_failed: {e}")
            return ctx
