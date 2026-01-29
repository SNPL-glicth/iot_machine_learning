"""Window management service.

Extracted from ml_features.py for modularity.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SensorWindow:
    """Sliding window for a single sensor."""
    sensor_id: int
    max_size: int
    max_age_seconds: float
    values: deque = field(default_factory=deque)
    timestamps: deque = field(default_factory=deque)
    
    def add(self, value: float, timestamp: float) -> None:
        """Add a new value to the window."""
        self.values.append(value)
        self.timestamps.append(timestamp)
        
        # Remove old values
        while len(self.values) > self.max_size:
            self.values.popleft()
            self.timestamps.popleft()
        
        # Remove old timestamps
        cutoff_time = time.time() - self.max_age_seconds
        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.timestamps.popleft()
            self.values.popleft()
    
    def get_values(self) -> List[float]:
        """Get all values in the window."""
        return list(self.values)
    
    def get_timestamps(self) -> List[float]:
        """Get all timestamps in the window."""
        return list(self.timestamps)
    
    @property
    def size(self) -> int:
        """Get current window size."""
        return len(self.values)
    
    @property
    def is_empty(self) -> bool:
        """Check if window is empty."""
        return len(self.values) == 0


class WindowManager:
    """Manages sliding windows for multiple sensors."""
    
    def __init__(self, max_size: int = 100, max_age_seconds: float = 300.0):
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self._windows: Dict[int, SensorWindow] = {}
    
    def add_reading(self, sensor_id: int, value: float, timestamp: Optional[float] = None) -> None:
        """Add a reading to the appropriate window."""
        if timestamp is None:
            timestamp = time.time()
        
        if sensor_id not in self._windows:
            self._windows[sensor_id] = SensorWindow(
                sensor_id=sensor_id,
                max_size=self.max_size,
                max_age_seconds=self.max_age_seconds,
            )
        
        self._windows[sensor_id].add(value, timestamp)
    
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
