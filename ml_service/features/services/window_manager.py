"""Window management service.

Extracted from ml_features.py for modularity.

FIX 2026-02-02: Agregada persistencia en Redis para evitar pérdida de contexto (ML-2).
"""

from __future__ import annotations

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SensorWindow:
    """Sliding window for a single sensor.
    
    FIX 2026-02-02: Optimizado con estadísticas incrementales Welford (P-4, P-5).
    """
    sensor_id: int
    max_size: int
    max_age_seconds: float
    values: deque = field(default_factory=deque)
    timestamps: deque = field(default_factory=deque)
    
    # Estadísticas incrementales (Welford's algorithm)
    _count: int = field(default=0, repr=False)
    _mean: float = field(default=0.0, repr=False)
    _m2: float = field(default=0.0, repr=False)  # Sum of squared differences
    
    # Cache para evitar list(deque) repetido
    _values_cache: Optional[tuple] = field(default=None, repr=False)
    _timestamps_cache: Optional[tuple] = field(default=None, repr=False)
    _cache_valid: bool = field(default=False, repr=False)
    
    def add(self, value: float, timestamp: float) -> None:
        """Add a new value to the window."""
        # Invalidar cache
        self._cache_valid = False
        self._values_cache = None
        self._timestamps_cache = None
        
        # Actualizar estadísticas incrementales (Welford)
        self._update_stats_add(value)
        
        self.values.append(value)
        self.timestamps.append(timestamp)
        
        # Remove old values
        while len(self.values) > self.max_size:
            removed = self.values.popleft()
            self.timestamps.popleft()
            self._update_stats_remove(removed)
        
        # Remove old timestamps
        cutoff_time = time.time() - self.max_age_seconds
        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.timestamps.popleft()
            removed = self.values.popleft()
            self._update_stats_remove(removed)
    
    def _update_stats_add(self, value: float) -> None:
        """Actualiza estadísticas al agregar un valor (Welford)."""
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2
    
    def _update_stats_remove(self, value: float) -> None:
        """Actualiza estadísticas al remover un valor."""
        if self._count <= 1:
            self._count = 0
            self._mean = 0.0
            self._m2 = 0.0
            return
        
        self._count -= 1
        delta = value - self._mean
        self._mean -= delta / self._count
        delta2 = value - self._mean
        self._m2 -= delta * delta2
        self._m2 = max(0.0, self._m2)  # Evitar negativos por errores de punto flotante
    
    def get_values(self) -> List[float]:
        """Get all values in the window (cached)."""
        if not self._cache_valid or self._values_cache is None:
            self._values_cache = tuple(self.values)
            self._timestamps_cache = tuple(self.timestamps)
            self._cache_valid = True
        return list(self._values_cache)
    
    def get_timestamps(self) -> List[float]:
        """Get all timestamps in the window (cached)."""
        if not self._cache_valid or self._timestamps_cache is None:
            self._values_cache = tuple(self.values)
            self._timestamps_cache = tuple(self.timestamps)
            self._cache_valid = True
        return list(self._timestamps_cache)
    
    @property
    def mean(self) -> float:
        """Media incremental O(1)."""
        return self._mean if self._count > 0 else 0.0
    
    @property
    def variance(self) -> float:
        """Varianza incremental O(1)."""
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)
    
    @property
    def std_dev(self) -> float:
        """Desviación estándar incremental O(1)."""
        return self.variance ** 0.5
    
    @property
    def size(self) -> int:
        """Get current window size."""
        return len(self.values)
    
    @property
    def is_empty(self) -> bool:
        """Check if window is empty."""
        return len(self.values) == 0


class WindowManager:
    """Manages sliding windows for multiple sensors.
    
    FIX 2026-02-02: Integra persistencia Redis para recuperar ventanas (ML-2).
    """
    
    def __init__(
        self,
        max_size: int = 100,
        max_age_seconds: float = 300.0,
        enable_persistence: bool = True,
    ):
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self._windows: Dict[int, SensorWindow] = {}
        self._enable_persistence = enable_persistence
        self._store = None
        
        if enable_persistence:
            try:
                from ..persistence import get_window_store
                self._store = get_window_store()
                if self._store.is_enabled:
                    logger.info("WindowManager: Redis persistence enabled")
            except Exception as e:
                logger.warning("WindowManager: Persistence disabled: %s", e)
    
    def add_reading(self, sensor_id: int, value: float, timestamp: Optional[float] = None) -> None:
        """Add a reading to the appropriate window."""
        if timestamp is None:
            timestamp = time.time()
        
        if sensor_id not in self._windows:
            # Intentar recuperar de Redis primero
            window = self._load_from_store(sensor_id)
            if window is None:
                window = SensorWindow(
                    sensor_id=sensor_id,
                    max_size=self.max_size,
                    max_age_seconds=self.max_age_seconds,
                )
            self._windows[sensor_id] = window
        
        self._windows[sensor_id].add(value, timestamp)
        
        # Persistir cada N lecturas para evitar overhead
        if self._windows[sensor_id].size % 10 == 0:
            self._save_to_store(sensor_id)
    
    def get_window(self, sensor_id: int) -> Optional[SensorWindow]:
        """Get the window for a sensor."""
        return self._windows.get(sensor_id)
    
    def get_all_sensor_ids(self) -> List[int]:
        """Get all sensor IDs being tracked."""
        return list(self._windows.keys())
    
    def remove_sensor(self, sensor_id: int) -> None:
        """Remove a sensor from tracking."""
        self._windows.pop(sensor_id, None)
    
    def clear_all(self) -> None:
        """Clear all windows."""
        self._windows.clear()
    
    def get_statistics(self) -> Dict[int, Dict]:
        """Get statistics for all sensors."""
        stats = {}
        for sensor_id, window in self._windows.items():
            stats[sensor_id] = {
                "size": window.size,
                "is_empty": window.is_empty,
                "oldest_timestamp": min(window.timestamps) if window.timestamps else None,
                "newest_timestamp": max(window.timestamps) if window.timestamps else None,
            }
        return stats
    
    def _load_from_store(self, sensor_id: int) -> Optional[SensorWindow]:
        """Carga una ventana desde Redis."""
        if not self._store or not self._store.is_enabled:
            return None
        
        try:
            persisted = self._store.load(sensor_id)
            if persisted is None:
                return None
            
            window = SensorWindow(
                sensor_id=sensor_id,
                max_size=self.max_size,
                max_age_seconds=self.max_age_seconds,
            )
            
            # Restaurar valores
            for v, t in zip(persisted.values, persisted.timestamps):
                window.values.append(v)
                window.timestamps.append(t)
            
            logger.info(
                "Restored window from Redis: sensor_id=%d values=%d",
                sensor_id, len(persisted.values)
            )
            return window
            
        except Exception as e:
            logger.warning("Failed to load window: sensor_id=%d error=%s", sensor_id, e)
            return None
    
    def _save_to_store(self, sensor_id: int) -> None:
        """Guarda una ventana en Redis."""
        if not self._store or not self._store.is_enabled:
            return
        
        window = self._windows.get(sensor_id)
        if window is None:
            return
        
        try:
            self._store.save(
                sensor_id=sensor_id,
                values=list(window.values),
                timestamps=list(window.timestamps),
            )
        except Exception as e:
            logger.warning("Failed to save window: sensor_id=%d error=%s", sensor_id, e)
    
    def persist_all(self) -> int:
        """Persiste todas las ventanas en Redis."""
        if not self._store or not self._store.is_enabled:
            return 0
        
        count = 0
        for sensor_id in self._windows:
            self._save_to_store(sensor_id)
            count += 1
        
        logger.info("Persisted %d windows to Redis", count)
        return count
