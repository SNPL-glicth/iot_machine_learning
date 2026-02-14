"""Port for sliding window stores.

Defines the contract that all sliding window implementations must follow.
This unifies the 5 different window implementations across the system:
1. SlidingWindowStore (ML consumer) — in-memory, per-sensor, LRU+TTL
2. SlidingWindowBuffer (ML service) — in-memory, per-sensor, stats
3. RedisWindowRepository (broker) — Redis ZSET, async
4. RedisWindowStore (ML features) — Redis JSON persistence
5. SensorWindow (ML domain) — value object container

E-15: Consolidate sliding window implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for a sliding window."""

    max_size: int = 100
    max_age_seconds: float = 3600.0
    max_sensors: int = 1000
    ttl_seconds: float = 3600.0


class ISlidingWindowStore(ABC, Generic[T]):
    """Interface for per-sensor sliding window stores.

    All implementations must support:
    - Appending items per sensor_id
    - Retrieving the window for a sensor_id
    - Clearing a sensor's window
    - Evicting stale/excess sensors
    - Reporting metrics
    """

    @abstractmethod
    def append(self, sensor_id: int, item: T, timestamp: float) -> int:
        """Append an item to a sensor's window.

        Args:
            sensor_id: Sensor identifier.
            item: The item to append.
            timestamp: Unix timestamp of the item.

        Returns:
            Current window size for this sensor.
        """

    @abstractmethod
    def get_window(self, sensor_id: int) -> List[T]:
        """Return items in the sensor's window, ordered by timestamp."""

    @abstractmethod
    def get_size(self, sensor_id: int) -> int:
        """Return number of items in the sensor's window."""

    @abstractmethod
    def clear(self, sensor_id: int) -> None:
        """Clear a sensor's window."""

    @abstractmethod
    def sensor_ids(self) -> List[int]:
        """Return all tracked sensor IDs."""

    @abstractmethod
    def evict_stale(self, current_time: float) -> int:
        """Evict sensors that have been idle beyond TTL.

        Returns:
            Number of sensors evicted.
        """

    @abstractmethod
    def get_metrics(self) -> dict:
        """Return occupancy and eviction metrics."""
