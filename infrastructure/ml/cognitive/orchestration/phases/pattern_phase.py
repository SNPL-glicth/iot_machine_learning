"""Pattern Phase — Pattern interpreter integration.

Integrates PatternInterpreter for human-readable pattern interpretation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import PipelineContext

try:
    from ...universal.analysis.pattern_interpreter import PatternInterpreter, InterpretedPattern
except (ImportError, ModuleNotFoundError):
    PatternInterpreter = None  # type: ignore[assignment,misc]
    InterpretedPattern = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class PatternPhase:
    """Phase: Pattern interpreter.
    
    Uses PatternInterpreter for human-readable interpretation of
    detected patterns with domain context and severity classification.
    """
    
    def __init__(self, pattern_interpreter: Optional[Any] = None) -> None:
        """Initialize pattern phase.
        
        Args:
            pattern_interpreter: Optional PatternInterpreter instance.
        """
        self._pattern_interpreter = pattern_interpreter
    
    @property
    def name(self) -> str:
        return "pattern"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute pattern phase.
        
        Args:
            ctx: Pipeline context with cognitive metrics.
        
        Returns:
            Updated context with interpreted patterns.
        """
        # Skip if no pattern interpreter available
        if self._pattern_interpreter is None or ctx.pattern_interpreter is None:
            return ctx
        
        try:
            # Use pattern interpreter from context if available
            pattern_interpreter = ctx.pattern_interpreter if ctx.pattern_interpreter else self._pattern_interpreter
            
            # Prepare pattern data for interpretation
            if ctx.values and ctx.profile:
                try:
                    # Create basic pattern data
                    pattern_data = {
                        "sensor_id": ctx.series_id,
                        "regime": ctx.regime,
                        "z_score": getattr(ctx.profile, "z_score", 0.0),
                        "trend": getattr(ctx.profile, "trend", "unknown"),
                        "confidence": ctx.fused_confidence or 0.0,
                    }
                    
                    # Interpret pattern
                    interpreted_pattern = pattern_interpreter.interpret(
                        pattern_data=pattern_data,
                        context={
                            "n_values": len(ctx.values),
                            "timestamps": ctx.timestamps,
                        },
                    )
                    
                    # Extract interpretation results
                    pattern_description = getattr(interpreted_pattern, 'description', '')
                    pattern_severity = getattr(interpreted_pattern, 'severity', 'unknown')
                    pattern_category = getattr(interpreted_pattern, 'category', 'unknown')
                    
                    # Log pattern interpretation summary
                    logger.debug(
                        "pattern_interpretation_completed",
                        extra={
                            "series_id": ctx.series_id,
                            "pattern_category": pattern_category,
                            "pattern_severity": pattern_severity,
                        },
                    )
                    
                    return ctx.with_field(
                        interpreted_pattern=interpreted_pattern,
                        pattern_description=pattern_description,
                        pattern_severity=pattern_severity,
                        pattern_category=pattern_category,
                    )
                except Exception as e:
                    logger.debug(f"pattern_interpretation_failed: {e}")
        
        except Exception as e:
            logger.debug(f"pattern_phase_failed: {e}")
        
        return ctx
