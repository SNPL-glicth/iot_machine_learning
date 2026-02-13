"""Re-export facade — backward compatibility.

Canonical location: ``domain.entities.iot.sensor_ranges``
"""

from .iot.sensor_ranges import DEFAULT_SENSOR_RANGES, get_default_range

__all__ = ["DEFAULT_SENSOR_RANGES", "get_default_range"]
