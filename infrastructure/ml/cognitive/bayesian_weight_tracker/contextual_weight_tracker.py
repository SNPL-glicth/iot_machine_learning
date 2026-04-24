"""Contextual Weight Tracker.

Tracks prediction errors by specific context for context-aware weight adaptation.
Uses modular ContextualErrorStorage for data management.
Renamed from 'plasticity' for honest naming.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from iot_machine_learning.domain.entities.plasticity.signal_context import SignalContext
from ..error_store import EngineErrorStore
from ..fusion.contextual_weight_calculator import resolve_contextual_weights
from .contextual_storage import ContextualErrorStorage

logger = logging.getLogger(__name__)


class ContextualWeightTracker:
    """Tracks MAE by context for adaptive weight calculation.
    
    Context key format: "{regime}|{volatility_binary}"
    Uses ContextualErrorStorage for thread-safe data management.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        min_samples: int = 5,
        epsilon: float = 0.1,
        max_contexts_per_engine: int = 20,
        error_store: Optional[EngineErrorStore] = None,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        if min_samples > window_size:
            raise ValueError(f"min_samples ({min_samples}) cannot exceed window_size ({window_size})")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if max_contexts_per_engine < 1:
            raise ValueError(f"max_contexts_per_engine must be >= 1, got {max_contexts_per_engine}")
        
        self._min_samples = min_samples
        self._epsilon = epsilon
        self._window_size = window_size  # Exposed for test compatibility
        self._error_store = error_store  # IMP-4a single-writer bus

        # Modular storage component — keeps context-keyed aggregation
        # (not a duplicate error log: the raw stream lives in EngineErrorStore).
        self._storage = ContextualErrorStorage(
            window_size=window_size,
            max_contexts_per_engine=max_contexts_per_engine,
        )
    
    @property
    def window_size(self) -> int:
        """Maximum errors to keep per context."""
        return self._window_size
    
    @property
    def min_samples(self) -> int:
        """Minimum samples required for weight calculation."""
        return self._min_samples
    
    @property
    def epsilon(self) -> float:
        """Small value to prevent division by zero."""
        return self._epsilon
    
    def _aggregate_context_key(self, context: SignalContext) -> str:
        """Aggregate context to reduce entropy from 288 to 32 buckets.
        
        Maps contexts to: {regime}|{time_block}|{volatility_binary}
        Matches SignalContext.context_key format.
        Result: Maximum 32 contexts per engine (4 regimes × 4 time blocks × 2 volatility).
        """
        time_block = context.time_of_day // 6
        vol_binary = "volatile" if context.volatility > 0.6 else "stable"
        return f"{context.regime.value}|{time_block}|{vol_binary}"
    
    def record_error(
        self,
        series_id: str,
        engine_name: str,
        error: float,
        context: SignalContext,
    ) -> None:
        """Record prediction error for a specific context.

        IMP-4a: when an :class:`EngineErrorStore` is injected the raw
        error is also forwarded to the store so it participates in the
        single-writer bus. The local context-keyed aggregation is kept
        because it indexes by context (not by raw error) — it is an
        aggregation, not a second log.
        """
        if error < 0:
            raise ValueError(f"error must be >= 0, got {error}")

        if self._error_store is not None:
            self._error_store.record(series_id, engine_name, error)

        context_key = self._aggregate_context_key(context)
        count = self._storage.record(series_id, engine_name, context_key, error)
        
        logger.debug(
            "contextual_error_recorded",
            extra={
                "series_id": series_id,
                "engine_name": engine_name,
                "context_key": context_key,
                "error": error,
                "count": count,
            },
        )
    
    def get_contextual_mae(
        self,
        series_id: str,
        engine_name: str,
        context: SignalContext,
    ) -> Optional[float]:
        """Get MAE for a specific context."""
        context_key = self._aggregate_context_key(context)
        errors = self._storage.get_errors(series_id, engine_name, context_key)
        
        if errors is None or len(errors) < self._min_samples:
            return None
        
        return sum(errors) / len(errors)
    
    def get_contextual_weights(
        self,
        series_id: str,
        engine_names: List[str],
        context: SignalContext,
    ) -> Optional[Dict[str, float]]:
        """Calculate contextual weights for engines."""
        if not engine_names:
            return {}
        
        maes = {
            name: self.get_contextual_mae(series_id, name, context)
            for name in engine_names
        }
        
        weights = resolve_contextual_weights(maes, engine_names, self._epsilon)
        
        if weights is not None:
            logger.debug(
                "contextual_weights_computed",
                extra={
                    "series_id": series_id,
                    "context_key": self._aggregate_context_key(context),
                    "maes": {k: v for k, v in maes.items() if v is not None},
                    "weights": weights,
                },
            )
        
        return weights
    
    def get_sample_count(
        self,
        series_id: str,
        engine_name: str,
        context: SignalContext,
    ) -> int:
        """Get number of samples for a specific context."""
        context_key = self._aggregate_context_key(context)
        return self._storage.count_samples(series_id, engine_name, context_key)
    
    def reset(
        self,
        series_id: Optional[str] = None,
        engine_name: Optional[str] = None,
    ) -> None:
        """Reset tracked errors."""
        self._storage.reset(series_id, engine_name)
    
    def get_all_contexts(
        self,
        series_id: str,
        engine_name: str,
    ) -> List[str]:
        """Get all context keys tracked for a series/engine."""
        return self._storage.get_all_contexts(series_id, engine_name)
