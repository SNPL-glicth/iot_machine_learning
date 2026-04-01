"""Iterative Cognitive Loop Controller.

Wraps the linear pipeline with confidence-based iteration.
Keeps executing until confidence threshold reached or max iterations.

Design principles:
- Wrap, don't modify: pipeline logic stays untouched
- Best-result tracking: keeps highest confidence result, not just last
- Time-bounded: respects budget to prevent runaway iteration
- Lightweight refine: minimal input modification between iterations
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .....domain.entities.prediction import Prediction

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class IterationConfig:
    """Configuration for iterative reasoning."""
    
    max_iterations: int = 3
    confidence_threshold: float = 0.85
    time_budget_ms: float = 5000.0
    
    # Refinement strategies (placeholders for future expansion)
    expand_window_on_retry: bool = False  # Placeholder: disabled for now
    window_expansion_factor: float = 1.5
    
    def __post_init__(self):
        """Validate config values."""
        if self.max_iterations < 1:
            self.max_iterations = 1
        if not 0.0 <= self.confidence_threshold <= 1.0:
            self.confidence_threshold = 0.85
        if self.time_budget_ms < 100.0:
            self.time_budget_ms = 100.0


@dataclass
class IterationRecord:
    """Record of a single iteration."""
    iteration: int
    confidence: float
    predicted_value: Optional[float]
    trend: str
    engine_used: str


class CognitiveLoopController:
    """Wraps pipeline execution with iterative confidence-based refinement.
    
    Usage:
        controller = CognitiveLoopController(
            pipeline_fn=execute_pipeline,
            config=IterationConfig(max_iterations=3, confidence_threshold=0.85)
        )
        result = controller.execute(orchestrator, values, timestamps, series_id)
    
    The controller will:
    1. Execute pipeline once
    2. Check if confidence >= threshold
    3. If not, refine input and retry (up to max_iterations)
    4. Return best result found across all iterations
    """
    
    def __init__(
        self,
        pipeline_fn: Callable,
        config: Optional[IterationConfig] = None,
    ):
        """Initialize controller.
        
        Args:
            pipeline_fn: Function to execute (typically execute_pipeline)
            config: Iteration configuration
        """
        self._pipeline = pipeline_fn
        self._config = config or IterationConfig()
        self._iteration_history: List[IterationRecord] = []
    
    def execute(
        self,
        orchestrator: Any,
        values: List[float],
        timestamps: Optional[List[float]],
        series_id: str,
        flags_snapshot: Optional[Any] = None,
    ) -> Prediction:
        """Execute with iterative refinement until confidence threshold or max iterations.
        
        Args:
            orchestrator: MetaCognitiveOrchestrator instance
            values: Time series values
            timestamps: Optional timestamps
            series_id: Series identifier
            flags_snapshot: Optional pre-captured feature flags
            
        Returns:
            result: Prediction, with best confidence found, annotated with iteration metadata
        """
        start_time_ms = time.time() * 1000
        
        best_result: Optional[Prediction] = None
        best_confidence = 0.0
        current_values = list(values)  # Copy to allow modification
        current_timestamps = timestamps.copy() if timestamps else None
        
        self._iteration_history = []
        
        for iteration in range(self._config.max_iterations):
            # Check time budget
            elapsed_ms = (time.time() * 1000) - start_time_ms
            if elapsed_ms >= self._config.time_budget_ms:
                logger.debug(
                    "cognitive_loop_budget_exceeded",
                    extra={
                        "series_id": series_id,
                        "iteration": iteration,
                        "elapsed_ms": elapsed_ms,
                        "budget_ms": self._config.time_budget_ms,
                    }
                )
                break
            
            # Execute pipeline
            try:
                result = self._pipeline(
                    orchestrator,
                    current_values,
                    current_timestamps,
                    series_id,
                    flags_snapshot=flags_snapshot,
                )
            except Exception as e:
                logger.exception(f"Pipeline execution failed iteration {iteration}: {e}")
                # Continue with next iteration if possible
                if best_result is not None:
                    break
                raise  # No best result yet, propagate error
            
            # Track this iteration
            record = IterationRecord(
                iteration=iteration,
                confidence=result.confidence,
                predicted_value=result.predicted_value,
                trend=result.trend,
                engine_used=result.metadata.get("selected_engine", "unknown"),
            )
            self._iteration_history.append(record)
            
            # Keep best result (by confidence)
            if result.confidence > best_confidence:
                best_result = result
                best_confidence = result.confidence
                logger.debug(
                    "cognitive_loop_new_best",
                    extra={
                        "series_id": series_id,
                        "iteration": iteration,
                        "confidence": best_confidence,
                    }
                )
            
            # Check convergence
            if best_confidence >= self._config.confidence_threshold:
                logger.info(
                    "cognitive_loop_converged",
                    extra={
                        "series_id": series_id,
                        "iterations": iteration + 1,
                        "final_confidence": best_confidence,
                        "threshold": self._config.confidence_threshold,
                    }
                )
                break
            
            # Refine for next iteration (placeholder - minimal modification)
            if iteration < self._config.max_iterations - 1:
                current_values, current_timestamps = self._refine(
                    current_values,
                    current_timestamps,
                    iteration,
                )
        
        # Annotate best result with iteration metadata
        if best_result is not None:
            # Ensure metadata dict exists
            if not hasattr(best_result, 'metadata') or best_result.metadata is None:
                best_result.metadata = {}
            
            best_result.metadata["cognitive_loop"] = {
                "iterations_used": len(self._iteration_history),
                "iteration_history": [
                    {
                        "iteration": r.iteration,
                        "confidence": r.confidence,
                        "predicted_value": r.predicted_value,
                        "trend": r.trend,
                        "engine_used": r.engine_used,
                    }
                    for r in self._iteration_history
                ],
                "converged": best_confidence >= self._config.confidence_threshold,
                "threshold": self._config.confidence_threshold,
                "time_budget_ms": self._config.time_budget_ms,
                "actual_time_ms": (time.time() * 1000) - start_time_ms,
            }
            
            return best_result
        
        # Should never happen if max_iterations >= 1, but for safety
        raise RuntimeError("Cognitive loop failed to produce any result")
    
    def _refine(
        self,
        values: List[float],
        timestamps: Optional[List[float]],
        iteration: int,
    ) -> tuple[List[float], Optional[List[float]]]:
        """Refine input for next iteration.
        
        Currently minimal placeholder - just returns input unchanged.
        Future: could expand window, smooth noise, etc.
        
        Args:
            values: Current values
            timestamps: Current timestamps
            iteration: Current iteration number (0-indexed)
            
        Returns:
            (refined_values, refined_timestamps)
        """
        # Placeholder: no refinement for now
        # Future implementations could:
        # - Fetch more historical data
        # - Apply smoothing
        # - Remove outliers
        # - etc.
        
        logger.debug(f"Refine placeholder iteration {iteration}: no changes")
        return values, timestamps
    
    def get_iteration_history(self) -> List[IterationRecord]:
        """Get history of last execution."""
        return self._iteration_history.copy()
