"""Dynamic Phase — Rolling window engine integration.

Integrates RollingWindowEngine for dynamic feature computation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import PipelineContext

try:
    from ...dynamic import RollingWindowEngine, DynamicFeaturePipeline
except (ImportError, ModuleNotFoundError):
    RollingWindowEngine = None  # type: ignore[assignment,misc]
    DynamicFeaturePipeline = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class DynamicPhase:
    """Phase: Dynamic feature computation.
    
    Uses RollingWindowEngine for dynamic feature computation with
    adaptive windows and context-aware feature extraction.
    """
    
    def __init__(self, rolling_window_engine: Optional[Any] = None) -> None:
        """Initialize dynamic phase.
        
        Args:
            rolling_window_engine: Optional RollingWindowEngine instance.
        """
        self._rolling_window_engine = rolling_window_engine
    
    @property
    def name(self) -> str:
        return "dynamic"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute dynamic phase.
        
        Args:
            ctx: Pipeline context with cognitive metrics.
        
        Returns:
            Updated context with dynamic features.
        """
        # Skip if no rolling window engine available
        if self._rolling_window_engine is None or ctx.rolling_window_engine is None:
            return ctx
        
        try:
            # Use rolling window engine from context if available
            rolling_window_engine = ctx.rolling_window_engine if ctx.rolling_window_engine else self._rolling_window_engine
            
            # Prepare data for dynamic feature computation
            if ctx.values and ctx.timestamps and len(ctx.values) > 10:
                try:
                    # Add current reading to rolling window
                    sensor_id = int(ctx.series_id) if ctx.series_id.isdigit() else 0
                    rolling_window_engine.add_reading(
                        sensor_id=sensor_id,
                        value=ctx.values[-1],
                        timestamp=ctx.timestamps[-1],
                    )
                    
                    # Compute dynamic features
                    dynamic_features = rolling_window_engine.compute_features(
                        sensor_id=sensor_id,
                        window_size=min(20, len(ctx.values)),
                    )
                    
                    # Extract key dynamic features
                    rolling_mean = dynamic_features.get('rolling_mean', 0.0)
                    rolling_std = dynamic_features.get('rolling_std', 0.0)
                    rolling_trend = dynamic_features.get('rolling_trend', 'stable')
                    
                    # Log dynamic feature summary
                    logger.debug(
                        "dynamic_features_computed",
                        extra={
                            "series_id": ctx.series_id,
                            "rolling_mean": rolling_mean,
                            "rolling_std": rolling_std,
                            "rolling_trend": rolling_trend,
                        },
                    )
                    
                    return ctx.with_field(
                        dynamic_features=dynamic_features,
                        rolling_mean=rolling_mean,
                        rolling_std=rolling_std,
                        rolling_trend=rolling_trend,
                    )
                except Exception as e:
                    logger.debug(f"dynamic_feature_computation_failed: {e}")
            else:
                logger.debug(
                    "dynamic_phase_skipped_insufficient_data",
                    extra={"series_id": ctx.series_id, "n_values": len(ctx.values) if ctx.values else 0},
                )
        
        except Exception as e:
            logger.debug(f"dynamic_phase_failed: {e}")
        
        return ctx
