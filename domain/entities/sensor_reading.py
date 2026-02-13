"""Re-export facade — backward compatibility.

Canonical location: ``domain.entities.iot.sensor_reading``
"""

from .iot.sensor_reading import SensorReading, SensorWindow

__all__ = ["SensorReading", "SensorWindow"]
