from __future__ import annotations

import logging
import time
from threading import RLock
from typing import Dict, List, Optional, Tuple

from ...domain.ports.series_correlation_port import SeriesCorrelationPort
from ...domain.services.interaction_field_service import InteractionFieldService

logger = logging.getLogger(__name__)


class SqlCorrelationAdapter(SeriesCorrelationPort):
    """SQL-based correlation adapter.

    get_correlated_series() reads ONLY from an in-memory cache that is
    populated externally by CorrelationBackgroundJob.  On a cache miss it
    returns [] immediately so the PERCEIVE phase is never blocked by SQL.

    refresh_series() is the entry-point for the background job to push
    fresh correlation data without blocking the prediction pipeline.
    """

    def __init__(
        self,
        storage_port,
        cache_ttl: int = 3600,
        min_correlation: float = 0.5,
        min_samples: int = 50,
    ) -> None:
        self._storage = storage_port
        self._cache_ttl = cache_ttl
        self._min_correlation = min_correlation
        self._min_samples = min_samples
        self._cache: Dict[str, List[Tuple[str, float]]] = {}
        self._cache_time: Dict[str, float] = {}
        self._lock = RLock()

    def get_correlated_series(
        self,
        series_id: str,
        max_neighbors: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return cached correlations — never blocks on SQL.

        Returns [] on cache miss.  The background job populates the cache
        periodically so the first few minutes after a cold start will have
        no correlation data, which is acceptable.
        """
        with self._lock:
            if series_id in self._cache:
                now = time.time()
                if now - self._cache_time.get(series_id, 0) < self._cache_ttl:
                    return self._cache[series_id][:max_neighbors]
        return []

    def refresh_series(self, series_id: str) -> None:
        """Compute and cache correlations for one series.

        Called by CorrelationBackgroundJob — never by the prediction pipeline.
        """
        correlations = self._compute_correlations(series_id)
        now = time.time()
        with self._lock:
            self._cache[series_id] = correlations
            self._cache_time[series_id] = now

    def get_recent_values_multi(
        self,
        series_ids: List[str],
        window: int,
    ) -> Dict[str, List[float]]:
        if not series_ids:
            return {}
        
        result: Dict[str, List[float]] = {}
        
        for sid in series_ids:
            try:
                sensor_id = int(sid) if sid.isdigit() else 0
                if sensor_id > 0:
                    sensor_window = self._storage.load_sensor_window(sensor_id, limit=window)
                    values = [r.value for r in sensor_window.readings]
                    if values:
                        result[sid] = values
            except Exception as e:
                logger.debug(f"get_recent_values_multi failed for {sid}: {e}")
                continue
        
        return result

    def _compute_correlations(self, series_id: str) -> List[Tuple[str, float]]:
        try:
            sensor_id = int(series_id) if series_id.isdigit() else 0
            if sensor_id == 0:
                return []
            
            target_window = self._storage.load_sensor_window(sensor_id, limit=1000)
            if len(target_window.readings) < self._min_samples:
                return []
            
            target_values = [r.value for r in target_window.readings]
            target_timestamps = [r.timestamp for r in target_window.readings]
            min_ts = min(target_timestamps)
            max_ts = max(target_timestamps)
            
            all_sensor_ids = self._storage.list_active_sensor_ids()
            correlations: List[Tuple[str, float]] = []
            
            for candidate_id in all_sensor_ids:
                if candidate_id == sensor_id:
                    continue
                
                try:
                    candidate_window = self._storage.load_sensor_window(candidate_id, limit=1000)
                    candidate_readings = [
                        r for r in candidate_window.readings
                        if min_ts <= r.timestamp <= max_ts
                    ]
                    
                    if len(candidate_readings) < self._min_samples:
                        continue
                    
                    aligned_pairs = self._align_readings(
                        target_timestamps, target_values,
                        [r.timestamp for r in candidate_readings],
                        [r.value for r in candidate_readings]
                    )
                    
                    if len(aligned_pairs) < self._min_samples:
                        continue
                    
                    corr = self._pearson_correlation(aligned_pairs)
                    
                    if abs(corr) >= self._min_correlation:
                        correlations.append((str(candidate_id), corr))
                
                except Exception as e:
                    logger.debug(f"correlation computation failed for candidate {candidate_id}: {e}")
                    continue
            
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            return correlations
        
        except Exception as e:
            logger.error(f"_compute_correlations failed for {series_id}: {e}")
            return []

    def _align_readings(
        self,
        ts1: List[float],
        vals1: List[float],
        ts2: List[float],
        vals2: List[float],
        tolerance: float = 60.0,
    ) -> List[Tuple[float, float]]:
        pairs: List[Tuple[float, float]] = []
        j = 0
        
        for i, t1 in enumerate(ts1):
            while j < len(ts2) and ts2[j] < t1 - tolerance:
                j += 1
            
            if j >= len(ts2):
                break
            
            if abs(ts2[j] - t1) <= tolerance:
                pairs.append((vals1[i], vals2[j]))
        
        return pairs

    def _pearson_correlation(self, pairs: List[Tuple[float, float]]) -> float:
        if len(pairs) < 2:
            return 0.0
        
        n = len(pairs)
        sum_x = sum(p[0] for p in pairs)
        sum_y = sum(p[1] for p in pairs)
        sum_xx = sum(p[0] * p[0] for p in pairs)
        sum_yy = sum(p[1] * p[1] for p in pairs)
        sum_xy = sum(p[0] * p[1] for p in pairs)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator_x = n * sum_xx - sum_x * sum_x
        denominator_y = n * sum_yy - sum_y * sum_y
        
        if denominator_x <= 0 or denominator_y <= 0:
            return 0.0
        
        denominator = (denominator_x * denominator_y) ** 0.5
        
        if denominator < 1e-12:
            return 0.0
        
        return numerator / denominator
    
    def smooth_with_field(
        self,
        predictions: Dict[str, float],
        smoothing_factor: float = 0.3,
    ) -> Dict[str, float]:
        """Apply Laplacian smoothing using interaction field.
        
        Optional method that uses InteractionFieldService to enforce
        consistency across correlated series. Falls back to original
        predictions if numpy unavailable or matrix singular.
        
        Args:
            predictions: Dict mapping series_id to prediction value
            smoothing_factor: Smoothing strength (0-1)
        
        Returns:
            Smoothed predictions dict (or original if smoothing fails)
        """
        try:
            series_ids = list(predictions.keys())
            if len(series_ids) < 2:
                return predictions
            
            try:
                import numpy as np
            except ImportError:
                return predictions
            
            corr_matrix = np.zeros((len(series_ids), len(series_ids)))
            for i, sid_i in enumerate(series_ids):
                for j, sid_j in enumerate(series_ids):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        with self._lock:
                            cached = self._cache.get(sid_i, [])
                        for neighbor_id, corr in cached:
                            if neighbor_id == sid_j:
                                corr_matrix[i, j] = corr
                                break
            
            field_service = InteractionFieldService(corr_matrix, series_ids)
            return field_service.smooth_predictions(predictions, smoothing_factor)
        except Exception:
            return predictions
    
    def get_cache_age(self, series_id: str) -> float:
        """Get cache age in seconds for a series.
        
        Args:
            series_id: Series identifier
        
        Returns:
            Age in seconds, or infinity if not cached
        """
        with self._lock:
            if series_id not in self._cache_time:
                return float('inf')
            return time.time() - self._cache_time[series_id]
