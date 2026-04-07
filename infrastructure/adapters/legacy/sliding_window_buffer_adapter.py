"""Adapter de compatibilidad para SlidingWindowBuffer legacy.

DEPRECADO. Delegando a SlidingWindowCache canónico.
Migrar a: infrastructure.persistence.sliding_window.SlidingWindowCache
"""

from __future__ import annotations

import warnings
from collections import deque
from math import sqrt
from typing import Dict, Iterable, Tuple

from iot_machine_learning.infrastructure.persistence.sliding_window import (
    SlidingWindowCache,
)


class WindowStats:
    """Estadísticos agregados para una ventana de tiempo."""
    
    def __init__(
        self,
        window_seconds: float,
        mean: float,
        min_val: float,
        max_val: float,
        std_dev: float,
        trend: float,
        count: int,
        last_value: float,
    ):
        self.window_seconds = window_seconds
        self.mean = mean
        self.min = min_val
        self.max = max_val
        self.std_dev = std_dev
        self.trend = trend
        self.count = count
        self.last_value = last_value


class SlidingWindowBuffer(SlidingWindowCache[Tuple[float, float]]):
    """DEPRECADO. Usa SlidingWindowCache directamente.
    
    Este adapter mantiene compatibilidad con código legacy que usa:
    - ml_service/sliding_window_buffer.SlidingWindowBuffer
    
    Diferencias con el original:
    - Usa series_id: str en lugar de sensor_id: int internamente
    - No recorta por max_horizon_seconds automáticamente
    """
    
    def __init__(
        self,
        max_horizon_seconds: float = 10.0,
        max_sensors: int = 1000,
        ttl_seconds: float = 3600.0,
    ) -> None:
        """Inicializa adapter con parámetros legacy."""
        warnings.warn(
            "SlidingWindowBuffer está deprecado. "
            "Usa infrastructure.persistence.sliding_window.SlidingWindowCache",
            DeprecationWarning,
            stacklevel=2,
        )
        
        self._max_horizon_seconds = float(max_horizon_seconds)
        
        # Delegar a canónico (window_size grande para horizon)
        super().__init__(
            window_size=10000,  # Grande para no limitar por tamaño
            max_series=max_sensors,
            ttl_seconds=int(ttl_seconds),
        )
    
    def add_reading(
        self,
        sensor_id: int,
        value: float,
        timestamp: float,
        windows: Iterable[float] = (1.0, 5.0, 10.0),
    ) -> Dict[str, WindowStats]:
        """Añade lectura y devuelve stats por ventana."""
        series_id = str(sensor_id)
        
        # Append to window
        self.append(series_id, (timestamp, value), timestamp)
        
        # Get all points
        items = self.get(series_id)
        if items is None:
            return {}
        
        # Recortar por horizon
        cutoff = timestamp - self._max_horizon_seconds
        buf = deque([(ts, val) for ts, (_, val) in items if ts >= cutoff])
        
        # Compute stats
        return self._compute_stats(buf, timestamp, windows)
    
    def _compute_stats(
        self,
        buf: deque,
        now_ts: float,
        windows: Iterable[float],
    ) -> Dict[str, WindowStats]:
        """Calcula estadísticos por ventana."""
        result: Dict[str, WindowStats] = {}
        if not buf:
            return result
        
        for w in windows:
            w = float(w)
            cutoff = now_ts - w
            window_points = [item for item in buf if item[0] >= cutoff]
            if not window_points:
                continue
            
            values = [v for (_, v) in window_points]
            v_min = min(values)
            v_max = max(values)
            v_mean = sum(values) / len(values)
            
            if len(values) >= 2:
                mean_diff_sq = [(v - v_mean) ** 2 for v in values]
                std_dev = sqrt(sum(mean_diff_sq) / len(values))
            else:
                std_dev = 0.0
            
            if len(window_points) >= 2:
                t0, v0 = window_points[0]
                t1, v1 = window_points[-1]
                dt = max(t1 - t0, 1e-3)
                trend = (v1 - v0) / dt
            else:
                trend = 0.0
            
            key = f"w{int(w)}"
            result[key] = WindowStats(
                window_seconds=w,
                mean=v_mean,
                min_val=v_min,
                max_val=v_max,
                std_dev=std_dev,
                trend=trend,
                count=len(values),
                last_value=values[-1],
            )
        
        return result
