"""Observability Phase — Cognitive observability metrics collection.

Collects and aggregates cognitive observability metrics for monitoring.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import PipelineContext

try:
    from ...observability import CognitiveObservabilityDashboard
except (ImportError, ModuleNotFoundError):
    CognitiveObservabilityDashboard = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class ObservabilityPhase:
    """Phase: Cognitive observability metrics collection.
    
    Aggregates cognitive metrics from the pipeline and provides
    observability insights for monitoring and alerting.
    """
    
    def __init__(self, observability_dashboard: Optional[Any] = None) -> None:
        """Initialize observability phase.
        
        Args:
            observability_dashboard: Optional CognitiveObservabilityDashboard instance.
        """
        self._observability_dashboard = observability_dashboard
        if CognitiveObservabilityDashboard is not None and self._observability_dashboard is None:
            self._observability_dashboard = CognitiveObservabilityDashboard()
    
    @property
    def name(self) -> str:
        return "observability"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute observability phase.
        
        Args:
            ctx: Pipeline context with cognitive metrics.
        
        Returns:
            Updated context with observability metrics.
        """
        # Skip if no observability dashboard available
        if self._observability_dashboard is None:
            return ctx
        
        try:
            # Collect observability metrics from context
            observability_metrics = self._collect_observability_metrics(ctx)
            
            # Get dashboard metrics
            dashboard_metrics = self._observability_dashboard.get_metrics()
            
            # Log observability summary
            logger.debug(
                "observability_metrics_collected",
                extra={
                    "series_id": ctx.series_id,
                    "regime": ctx.regime,
                    "fused_confidence": ctx.fused_confidence,
                    "drift_detected": ctx.metadata.get("drift_detected", False),
                },
            )
            
            return ctx.with_field(
                observability_metrics=observability_metrics,
                dashboard_metrics=dashboard_metrics,
            )
        
        except Exception as e:
            logger.debug(f"observability_phase_failed: {e}")
            return ctx
    
    def _collect_observability_metrics(self, ctx: PipelineContext) -> dict:
        """Collect observability metrics from pipeline context.
        
        Args:
            ctx: Pipeline context.
        
        Returns:
            Dictionary with observability metrics.
        """
        metrics = {
            "series_id": ctx.series_id,
            "regime": ctx.regime,
            "fused_confidence": ctx.fused_confidence,
            "fused_value": ctx.fused_value,
            "selected_engine": ctx.selected_engine,
            "n_engines_active": len([p for p in (ctx.perceptions or []) 
                                    if not getattr(p, 'inhibited', False)]),
            "n_engines_inhibited": len([p for p in (ctx.perceptions or []) 
                                       if getattr(p, 'inhibited', False)]),
            "drift_detected": ctx.metadata.get("drift_detected", False),
            "drift_score": ctx.metadata.get("drift_score", 0.0),
            "pipeline_duration_ms": ctx.timer.total_ms if ctx.timer else 0.0,
            "is_fallback": ctx.is_fallback,
            "fallback_reason": ctx.fallback_reason,
        }
        
        # Add validation metrics if available
        if hasattr(ctx, 'validation_result') and ctx.validation_result:
            metrics["explainability_quality"] = ctx.validation_result.get("explainability_quality_score", 0.0)
            metrics["temporal_consistency"] = ctx.validation_result.get("temporal_consistency", 0.0)
        
        return metrics
