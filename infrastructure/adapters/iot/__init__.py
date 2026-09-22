"""Adapters IoT para la plataforma Zenin.

Provee adaptadores planos y desacoplados:
- Sensores y traducción de series temporales (sensor_adapter.py)
- Persistencia de estados ML Redis e In-Memory (ml_state_store_adapter.py)
- Consultas de umbrales SQL (severity_sql_adapter.py)
- Despacho y manejo de actuadores (actuator_adapter.py)
- Detección de deriva de telemetría (drift_adapter.py)
"""

from __future__ import annotations

from .actuator_adapter import (
    ActuatorCommand,
    ActuatorConfig,
    CallbackActuatorClient,
    IoTActuatorHandler,
    MockActuatorClient,
)
from .drift_adapter import (
    DriftSensorAdapter,
    IoTDriftSensorAdapter,
    create_drift_detector,
)
from .ml_state_store_adapter import (
    InMemoryMLStateStore,
    RedisMLStateStore,
    create_state_store,
)
from .sensor_adapter import (
    sensor_id_to_series_id,
    sensor_reading_to_data_point,
    sensor_readings_to_time_window,
)
from .severity_sql_adapter import SeveritySqlAdapter

__all__ = [
    # Sensores
    "sensor_id_to_series_id",
    "sensor_reading_to_data_point",
    "sensor_readings_to_time_window",
    # Persistencia
    "InMemoryMLStateStore",
    "RedisMLStateStore",
    "create_state_store",
    "SeveritySqlAdapter",
    # Actuadores
    "ActuatorConfig",
    "ActuatorCommand",
    "MockActuatorClient",
    "CallbackActuatorClient",
    "IoTActuatorHandler",
    # Drift
    "create_drift_detector",
    "IoTDriftSensorAdapter",
    "DriftSensorAdapter",
]
