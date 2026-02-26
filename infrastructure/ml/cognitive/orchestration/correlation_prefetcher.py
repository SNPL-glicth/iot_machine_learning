from __future__ import annotations

import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CorrelationPrefetcher:
    """Asynchronously prefetches neighbor values to avoid blocking PERCEIVE phase."""
    
    def __init__(
        self,
        correlation_port,
        cache_ttl: float = 30.0,
        max_cache: int = 1000,
        max_workers: int = 2,
    ):
        self._correlation_port = correlation_port
        self._cache_ttl = cache_ttl
        self._max_cache = max_cache
        self._cache: OrderedDict[str, Tuple[List, Dict, float]] = OrderedDict()
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="corr-prefetch"
        )
    
    def prefetch_async(self, series_id: str) -> None:
        """Non-blocking prefetch of neighbors."""
        if not self._correlation_port:
            return
        self._executor.submit(self._fetch_and_cache, series_id)
    
    def _fetch_and_cache(self, series_id: str):
        """Fetch neighbors and cache them."""
        try:
            neighbors = self._correlation_port.get_correlated_series(series_id, max_neighbors=3)
            if neighbors:
                neighbor_ids = [n[0] for n in neighbors]
                neighbor_values = self._correlation_port.get_recent_values_multi(neighbor_ids, window=5)
                
                with self._lock:
                    self._cache[series_id] = (neighbors, neighbor_values, time.time())
                    if len(self._cache) > self._max_cache:
                        self._cache.popitem(last=False)
        except Exception as e:
            logger.debug(f"prefetch_failed series={series_id}: {e}")
    
    def get_cached(self, series_id: str) -> Optional[Tuple[List, Dict]]:
        """Get cached neighbors (non-blocking)."""
        with self._lock:
            if series_id in self._cache:
                neighbors, values, timestamp = self._cache[series_id]
                if time.time() - timestamp < self._cache_ttl:
                    return neighbors, values
        return None
    
    def shutdown(self):
        """Shutdown executor."""
        self._executor.shutdown(wait=False)
