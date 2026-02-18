"""Contextual Plasticity Tracker.

Tracks prediction errors by specific context (regime + time + volatility)
for context-aware weight adaptation.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from threading import RLock
from typing import Dict, List, Optional

from ....domain.entities.plasticity.plasticity_context import PlasticityContext
from .contextual_weight_calculator import resolve_contextual_weights

logger = logging.getLogger(__name__)


class ContextualPlasticityTracker:
    """Tracks MAE by context for adaptive weight calculation.
    
    Structure: {series_id: {engine_name: {context_key: [errors]}}}
    
    Context key format: "{regime}|{hour}|{volatility_bucket}"
    
    Thread Safety:
        Uses RLock to protect shared dictionary from race conditions.
        RLock allows re-entrance from the same thread (prevents deadlocks).
    
    Attributes:
        window_size: Maximum errors to keep per context (default: 50)
        min_samples: Minimum samples required for weight calculation (default: 5)
        epsilon: Small value to prevent division by zero (default: 0.1)
    
    Examples:
        >>> tracker = ContextualPlasticityTracker()
        >>> ctx = PlasticityContext.create_default(RegimeType.STABLE)
        >>> tracker.record_error("sensor_1", "taylor", 5.0, ctx)
        >>> weights = tracker.get_contextual_weights("sensor_1", ["taylor"], ctx)
        >>> weights is None  # Not enough samples yet
        True
    """
    
    def __init__(
        self,
        window_size: int = 50,
        min_samples: int = 5,
        epsilon: float = 0.1,
    ) -> None:
        """Initialize contextual plasticity tracker.
        
        Args:
            window_size: Maximum errors to keep per context
            min_samples: Minimum samples required for weight calculation
            epsilon: Small value to prevent division by zero
        
        Raises:
            ValueError: If parameters are invalid
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        if min_samples > window_size:
            raise ValueError(f"min_samples ({min_samples}) cannot exceed window_size ({window_size})")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        
        self.window_size = window_size
        self.min_samples = min_samples
        self.epsilon = epsilon
        
        # Structure: {series_id: {engine_name: {context_key: deque[errors]}}}
        self._errors: Dict[str, Dict[str, Dict[str, deque]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: deque(maxlen=window_size)))
        )
        
        # Thread safety: RLock for protecting shared dictionary
        self._lock = RLock()
    
    def record_error(
        self,
        series_id: str,
        engine_name: str,
        error: float,
        context: PlasticityContext,
    ) -> None:
        """Record prediction error for a specific context (thread-safe).
        
        Args:
            series_id: Series identifier
            engine_name: Name of the prediction engine
            error: Absolute prediction error
            context: Plasticity context
        
        Raises:
            ValueError: If error is negative
        """
        if error < 0:
            raise ValueError(f"error must be >= 0, got {error}")
        
        context_key = context.context_key
        
        with self._lock:
            self._errors[series_id][engine_name][context_key].append(error)
        
        logger.debug(
            "contextual_error_recorded",
            extra={
                "series_id": series_id,
                "engine_name": engine_name,
                "context_key": context_key,
                "error": error,
                "count": len(self._errors[series_id][engine_name][context_key]),
            },
        )
    
    def get_contextual_mae(
        self,
        series_id: str,
        engine_name: str,
        context: PlasticityContext,
    ) -> Optional[float]:
        """Get MAE for a specific context (thread-safe).
        
        Args:
            series_id: Series identifier
            engine_name: Name of the prediction engine
            context: Plasticity context
        
        Returns:
            MAE if enough samples, None otherwise
        """
        context_key = context.context_key
        
        with self._lock:
            if series_id not in self._errors:
                return None
            if engine_name not in self._errors[series_id]:
                return None
            if context_key not in self._errors[series_id][engine_name]:
                return None
            
            errors = self._errors[series_id][engine_name][context_key]
            
            if len(errors) < self.min_samples:
                return None
            
            mae = sum(errors) / len(errors)
            return mae
    
    def get_contextual_weights(
        self,
        series_id: str,
        engine_names: List[str],
        context: PlasticityContext,
    ) -> Optional[Dict[str, float]]:
        """Calculate contextual weights for engines (thread-safe).
        
        Formula: weight = 1 / (mae + epsilon), normalized to sum=1.0.
        Returns None if any engine lacks sufficient data.
        
        Args:
            series_id: Series identifier
            engine_names: List of engine names
            context: Plasticity context
        
        Returns:
            Dict of normalized weights, or None if insufficient data
        """
        if not engine_names:
            return {}

        maes: Dict[str, Optional[float]] = {
            name: self.get_contextual_mae(series_id, name, context)
            for name in engine_names
        }

        weights = resolve_contextual_weights(maes, engine_names, self.epsilon)

        if weights is not None:
            logger.debug(
                "contextual_weights_computed",
                extra={
                    "series_id": series_id,
                    "context_key": context.context_key,
                    "maes": {k: v for k, v in maes.items() if v is not None},
                    "weights": weights,
                },
            )

        return weights
    
    def get_sample_count(
        self,
        series_id: str,
        engine_name: str,
        context: PlasticityContext,
    ) -> int:
        """Get number of samples for a specific context.
        
        Args:
            series_id: Series identifier
            engine_name: Name of the prediction engine
            context: Plasticity context
        
        Returns:
            Number of samples
        """
        context_key = context.context_key
        
        if series_id not in self._errors:
            return 0
        if engine_name not in self._errors[series_id]:
            return 0
        if context_key not in self._errors[series_id][engine_name]:
            return 0
        
        return len(self._errors[series_id][engine_name][context_key])
    
    def reset(
        self,
        series_id: Optional[str] = None,
        engine_name: Optional[str] = None,
    ) -> None:
        """Reset tracked errors (thread-safe).
        
        Args:
            series_id: If provided, reset only this series
            engine_name: If provided, reset only this engine (requires series_id)
        """
        with self._lock:
            if series_id is None:
                self._errors.clear()
                logger.info("contextual_tracker_reset_all")
            elif engine_name is None:
                if series_id in self._errors:
                    del self._errors[series_id]
                    logger.info("contextual_tracker_reset_series", extra={"series_id": series_id})
            else:
                if series_id in self._errors and engine_name in self._errors[series_id]:
                    del self._errors[series_id][engine_name]
                    logger.info(
                        "contextual_tracker_reset_engine",
                        extra={"series_id": series_id, "engine_name": engine_name},
                    )
    
    def get_all_contexts(
        self,
        series_id: str,
        engine_name: str,
    ) -> List[str]:
        """Get all context keys tracked for a series/engine (thread-safe).
        
        Args:
            series_id: Series identifier
            engine_name: Name of the prediction engine
        
        Returns:
            List of context keys
        """
        with self._lock:
            return list(self._errors.get(series_id, {}).get(engine_name, {}).keys())
