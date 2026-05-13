"""Adapt Phase — MED-1 Refactoring.

Weight resolution and plasticity adaptation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import PipelineContext

logger = logging.getLogger(__name__)


class WeightCache:
    """LRU cache for plasticity weights."""
    
    def __init__(self, max_entries: int = 1000, ttl_seconds: float = 60.0):
        from collections import OrderedDict
        from threading import RLock
        import time
        
        self._cache: OrderedDict = OrderedDict()
        self._lock = RLock()
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._time = time
    
    def get(self, series_id: str, regime: str):
        with self._lock:
            from typing import Tuple, Dict
            key = (series_id, regime)
            if key not in self._cache:
                return None
            weights, timestamp = self._cache[key]
            if self._time.time() - timestamp >= self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return weights
    
    def set(self, series_id: str, regime: str, weights):
        with self._lock:
            import time
            key = (series_id, regime)
            while len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)
            self._cache[key] = (weights, time.time())


# CRIT-1 FIX: Removed global _weight_cache singleton.
# WeightCache is now instance-scoped within AdaptPhase.__init__()
# to eliminate race conditions in concurrent pipeline executions.


class AdaptPhase:
    """Phase 3: Plasticity weight resolution.
    
    CRIT-1 FIX: Instance-scoped weight cache instead of global singleton.
    Each AdaptPhase instance has its own cache, eliminating race conditions
    in concurrent pipeline executions.
    """
    
    def __init__(self, max_cache_entries: int = 1000, cache_ttl_seconds: float = 60.0) -> None:
        """Initialize AdaptPhase with instance-scoped weight cache.
        
        Args:
            max_cache_entries: Maximum number of cached weight entries
            cache_ttl_seconds: Time-to-live for cache entries
        """
        self._weight_cache = WeightCache(
            max_entries=max_cache_entries,
            ttl_seconds=cache_ttl_seconds
        )
        logger.debug(
            "adapt_phase_initialized",
            extra={
                "max_entries": max_cache_entries,
                "ttl_seconds": cache_ttl_seconds,
            },
        )
    
    @property
    def name(self) -> str:
        return "adapt"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute adaptation phase."""
        orchestrator = ctx.orchestrator
        
        # Get error dict scoped to series_id (CRIT-1)
        engine_names = [p.engine_name for p in ctx.perceptions]
        error_dict = orchestrator._error_history.get_error_dict_for_inhibition(
            ctx.series_id, engine_names
        )
        
        # Resolve plasticity weights using instance cache
        plasticity_weights = self._weight_cache.get(ctx.series_id, ctx.regime)
        if plasticity_weights is None:
            plasticity_weights = orchestrator._weight_resolver.resolve(
                regime=ctx.regime,
                engine_names=engine_names,
                series_id=ctx.series_id,
            )
            self._weight_cache.set(ctx.series_id, ctx.regime, plasticity_weights)
            logger.debug(
                "adapt_phase_cache_miss",
                extra={
                    "series_id": ctx.series_id,
                    "regime": ctx.regime,
                    "cache_size": len(self._weight_cache._cache),
                },
            )
        else:
            logger.debug(
                "adapt_phase_cache_hit",
                extra={
                    "series_id": ctx.series_id,
                    "regime": ctx.regime,
                },
            )
        
        # Track adaptation status
        adapted = (
            orchestrator._plasticity is not None and 
            orchestrator._plasticity.has_history(ctx.regime)
        )
        
        # Update explanation builder if available
        if ctx.explanation and hasattr(ctx.explanation, 'set_adaptation'):
            ctx.explanation.set_adaptation(adapted=adapted, regime=ctx.regime)
        
        return ctx.with_field(
            error_dict=error_dict,
            plasticity_weights=plasticity_weights,
        )
