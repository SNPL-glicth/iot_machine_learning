"""Neural Phase — Hybrid neural network integration.

Integrates HybridNeuralEngine for advanced pattern detection and prediction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import PipelineContext

try:
    from ...neural import HybridNeuralEngine, NeuralResult
except (ImportError, ModuleNotFoundError):
    HybridNeuralEngine = None  # type: ignore[assignment,misc]
    NeuralResult = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class NeuralPhase:
    """Phase: Hybrid neural network analysis.
    
    Uses HybridNeuralEngine for advanced pattern detection,
    non-linear prediction, and neuromorphic analysis.
    """
    
    def __init__(self, neural_engine: Optional[Any] = None) -> None:
        """Initialize neural phase.
        
        Args:
            neural_engine: Optional HybridNeuralEngine instance.
        """
        self._neural_engine = neural_engine
    
    @property
    def name(self) -> str:
        return "neural"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute neural phase.
        
        Args:
            ctx: Pipeline context with cognitive metrics.
        
        Returns:
            Updated context with neural analysis results.
        """
        # Skip if no neural engine available
        if self._neural_engine is None or ctx.neural_engine is None:
            return ctx
        
        try:
            # Use neural engine from context if available
            neural_engine = ctx.neural_engine if ctx.neural_engine else self._neural_engine
            
            # Prepare input for neural engine
            if ctx.values and len(ctx.values) > 10:
                try:
                    # Run neural analysis
                    neural_result = neural_engine.analyze(
                        values=ctx.values,
                        timestamps=ctx.timestamps,
                        series_id=ctx.series_id,
                    )
                    
                    # Extract neural predictions and confidence
                    neural_prediction = getattr(neural_result, 'predicted_value', None)
                    neural_confidence = getattr(neural_result, 'confidence', 0.0)
                    neural_energy = getattr(neural_result, 'energy_consumption', 0.0)
                    
                    # Log neural analysis summary
                    logger.debug(
                        "neural_analysis_completed",
                        extra={
                            "series_id": ctx.series_id,
                            "neural_prediction": neural_prediction,
                            "neural_confidence": neural_confidence,
                            "energy_consumption": neural_energy,
                        },
                    )
                    
                    return ctx.with_field(
                        neural_result=neural_result,
                        neural_prediction=neural_prediction,
                        neural_confidence=neural_confidence,
                        neural_energy=neural_energy,
                    )
                except Exception as e:
                    logger.debug(f"neural_analysis_failed: {e}")
            else:
                logger.debug(
                    "neural_phase_skipped_insufficient_data",
                    extra={"series_id": ctx.series_id, "n_values": len(ctx.values) if ctx.values else 0},
                )
        
        except Exception as e:
            logger.debug(f"neural_phase_failed: {e}")
        
        return ctx
