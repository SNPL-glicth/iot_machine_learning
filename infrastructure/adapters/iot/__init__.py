"""Adapters para traducción IoT legacy → Zenin canónico."""

from .sensor_adapter import (
    sensor_id_to_series_id,
    sensor_reading_to_data_point,
    sensor_readings_to_time_window,
)

__all__ = [
    "sensor_id_to_series_id",
    "sensor_reading_to_data_point",
    "sensor_readings_to_time_window",
]
