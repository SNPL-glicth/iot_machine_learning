"""Re-export facade — backward compatibility.

Canonical location: ``domain.entities.iot.sensor_reading``
"""

from .iot.sensor_reading import Reading, SensorReading, SensorWindow, TimeSeriesWindow

__all__ = ["Reading", "SensorReading", "SensorWindow", "TimeSeriesWindow"]
