"""
RollingWindowEngine for managing multiple rolling windows per sensor.

Maintains multiple rolling windows (1h, 6h, 24h) for efficient computation
of rolling statistics.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import time
from threading import RLock

from .models.rolling_stats import RollingStats


@dataclass
class SensorRollingWindow:
    """Rolling window for a single sensor and window size."""
    sensor_id: int
    size_minutes: int
    max_size: int
    max_age_seconds: float
    
    def __post_init__(self):
        self.values: deque = deque(maxlen=self.max_size)
        self.timestamps: deque = deque(maxlen=self.max_size)
        self._count = 0
        self._sum = 0.0
        self._sum_sq = 0.0
    
    def add(self, value: float, timestamp: float) -> None:
        """Add a value to the window."""
        # Remove old values if deque is full
        if len(self.values) == self.max_size:
            old_value = self.values[0]
            old_timestamp = self.timestamps[0]
            self._sum -= old_value
            self._sum_sq -= old_value * old_value
            self._count -= 1
        
        self.values.append(value)
        self.timestamps.append(timestamp)
        self._sum += value
        self._sum_sq += value * value
        self._count += 1
    
    def cleanup_old(self, current_timestamp: float) -> None:
        """Remove values older than max_age_seconds."""
        while self.timestamps and (current_timestamp - self.timestamps[0]) > self.max_age_seconds:
            old_value = self.values.popleft()
            old_timestamp = self.timestamps.popleft()
            self._sum -= old_value
            self._sum_sq -= old_value * old_value
            self._count -= 1
    
    def get_stats(self) -> Optional[RollingStats]:
        """Compute rolling statistics."""
        if self._count == 0:
            return None
        
        mean = self._sum / self._count
        variance = (self._sum_sq / self._count) - (mean * mean)
        std = variance ** 0.5 if variance > 0 else 0.0
        
        min_val = min(self.values) if self.values else 0.0
        max_val = max(self.values) if self.values else 0.0
        
        return RollingStats(
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            count=self._count,
        )
    
    def is_empty(self) -> bool:
        """Check if window is empty."""
        return self._count == 0
    
    def get_values(self) -> List[float]:
        """Get all values as list."""
        return list(self.values)
    
    def get_timestamps(self) -> List[float]:
        """Get all timestamps as list."""
        return list(self.timestamps)


class RollingWindowEngine:
    """Engine for managing multiple rolling windows per sensor."""
    
    def __init__(
        self,
        window_sizes_minutes: List[int] = None,
        enable_persistence: bool = False,
        max_sensor_age_seconds: float = 86400.0,
        max_cache_age_seconds: float = 60.0,
    ):
        """
        Initialize rolling window engine.
        
        Args:
            window_sizes_minutes: List of window sizes in minutes (default: [60, 360, 1440])
            enable_persistence: Whether to enable persistence (future feature)
            max_sensor_age_seconds: Max age before evicting inactive sensor windows
            max_cache_age_seconds: Max age for cached stats
        """
        self._window_sizes = window_sizes_minutes or [60, 360, 1440]
        self._enable_persistence = enable_persistence
        self._max_sensor_age_seconds = max_sensor_age_seconds
        self._max_cache_age_seconds = max_cache_age_seconds
        
        # sensor_id -> size_minutes -> SensorRollingWindow
        self._windows: Dict[int, Dict[int, SensorRollingWindow]] = {}
        self._sensor_last_updated: Dict[int, float] = {}
        
        # Cache for stats to avoid recomputation
        self._stats_cache: Dict[int, Dict[int, RollingStats]] = {}
        self._cache_timestamps: Dict[int, Dict[int, float]] = {}
        
        # Per-sensor locks for thread-safe concurrent access
        self._sensor_locks: Dict[int, RLock] = defaultdict(RLock)
        # Global lock for registry-level operations (sensor iteration, cleanup)
        self._lock = RLock()
    
    def add_reading(self, sensor_id: int, value: float, timestamp: Optional[float] = None) -> None:
        """
        Add a reading to all windows for the sensor.
        
        Args:
            sensor_id: Sensor identifier
            value: Sensor value
            timestamp: Unix timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._sensor_locks[sensor_id]:
            # Initialize windows for sensor if needed
            if sensor_id not in self._windows:
                self._windows[sensor_id] = {}
                self._sensor_last_updated[sensor_id] = timestamp
            
            # Update last updated timestamp
            self._sensor_last_updated[sensor_id] = timestamp
            
            # Add to each window
            for size_minutes in self._window_sizes:
                if size_minutes not in self._windows[sensor_id]:
                    max_size = size_minutes * 60  # 1 point per second
                    max_age_seconds = size_minutes * 60
                    self._windows[sensor_id][size_minutes] = SensorRollingWindow(
                        sensor_id=sensor_id,
                        size_minutes=size_minutes,
                        max_size=max_size,
                        max_age_seconds=max_age_seconds,
                    )
                
                window = self._windows[sensor_id][size_minutes]
                window.cleanup_old(timestamp)
                window.add(value, timestamp)
                
                # Invalidate cache for this window
                if sensor_id in self._stats_cache:
                    self._stats_cache[sensor_id].pop(size_minutes, None)
                    self._cache_timestamps[sensor_id].pop(size_minutes, None)
    
    def compute_stats(
        self,
        sensor_id: int,
        window_sizes: Optional[List[int]] = None,
    ) -> Dict[int, RollingStats]:
        """
        Compute rolling statistics for specified windows.
        
        Args:
            sensor_id: Sensor identifier
            window_sizes: List of window sizes in minutes (default: all configured windows)
        
        Returns:
            Dictionary mapping window_size -> RollingStats
        """
        with self._sensor_locks[sensor_id]:
            self._cleanup_old_sensors()
            self._cleanup_old_cache()
            
            sizes = window_sizes or self._window_sizes
            stats = {}
            
            if sensor_id not in self._windows:
                return stats
            
            for size in sizes:
                window = self._windows[sensor_id].get(size)
                if window and not window.is_empty:
                    stats[size] = window.get_stats()
            
            return stats
    
    def get_window(
        self,
        sensor_id: int,
        size_minutes: int,
    ) -> Optional[SensorRollingWindow]:
        """
        Get a specific rolling window.
        
        Args:
            sensor_id: Sensor identifier
            size_minutes: Window size in minutes
        
        Returns:
            SensorRollingWindow or None if not found
        """
        with self._sensor_locks[sensor_id]:
            return self._windows.get(sensor_id, {}).get(size_minutes)
    
    def get_values(
        self,
        sensor_id: int,
        size_minutes: int,
    ) -> List[float]:
        """
        Get all values from a specific window.
        
        Args:
            sensor_id: Sensor identifier
            size_minutes: Window size in minutes
        
        Returns:
            List of values
        """
        window = self.get_window(sensor_id, size_minutes)
        return window.get_values() if window else []
    
    def get_timestamps(
        self,
        sensor_id: int,
        size_minutes: int,
    ) -> List[float]:
        """
        Get all timestamps from a specific window.
        
        Args:
            sensor_id: Sensor identifier
            size_minutes: Window size in minutes
        
        Returns:
            List of timestamps
        """
        window = self.get_window(sensor_id, size_minutes)
        return window.get_timestamps() if window else []
    
    def cleanup_sensor(self, sensor_id: int) -> None:
        """
        Remove all windows for a sensor (for cleanup).
        
        Args:
            sensor_id: Sensor identifier
        """
        with self._sensor_locks[sensor_id]:
            self._windows.pop(sensor_id, None)
            self._sensor_last_updated.pop(sensor_id, None)
            self._stats_cache.pop(sensor_id, None)
            self._cache_timestamps.pop(sensor_id, None)
    
    def _cleanup_old_sensors(self) -> None:
        """Clean up sensor windows that haven't been updated recently."""
        current_time = time.time()
        cutoff_time = current_time - self._max_sensor_age_seconds
        
        sensors_to_remove = []
        with self._lock:
            for sensor_id, last_updated in self._sensor_last_updated.items():
                if last_updated < cutoff_time:
                    sensors_to_remove.append(sensor_id)
        
        for sensor_id in sensors_to_remove:
            self.cleanup_sensor(sensor_id)
    
    def _cleanup_old_cache(self) -> None:
        """Clean up old cache entries."""
        current_time = time.time()
        cutoff_time = current_time - self._max_cache_age_seconds
        
        with self._lock:
            for sensor_id in list(self._cache_timestamps.keys()):
                for size in list(self._cache_timestamps.get(sensor_id, {}).keys()):
                    if self._cache_timestamps[sensor_id][size] < cutoff_time:
                        self._stats_cache.get(sensor_id, {}).pop(size, None)
                        self._cache_timestamps.get(sensor_id, {}).pop(size, None)
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Estimate memory usage.
        
        Returns:
            Dictionary with memory usage statistics
        """
        with self._lock:
            total_sensors = len(self._windows)
            total_windows = sum(len(windows) for windows in self._windows.values())
            total_points = sum(
                window._count
                for windows in self._windows.values()
                for window in windows.values()
            )
        
        return {
            "total_sensors": total_sensors,
            "total_windows": total_windows,
            "total_points": total_points,
            "estimated_bytes": total_points * 16,  # 8 bytes for value + 8 for timestamp
        }
