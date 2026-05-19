"""FIX P3-4: Configuración de features por sensor o por tipo de sensor.

Variables de entorno:
  ML_FEATURE_CONFIG_BY_TYPE (default: "{}")
  ML_FEATURE_CONFIG_BY_ID   (default: "{}")
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Defaults globales (retrocompatibles con feature_computer.py)
DEFAULT_MIN_WINDOW_AGE = 30.0
DEFAULT_MIN_READINGS = 5
DEFAULT_MAX_WINDOW_SIZE = 100
DEFAULT_SAMPLING_HZ = 1.0


@dataclass(frozen=True)
class SensorFeatureConfig:
    """Configuración de features por sensor o tipo."""
    min_window_age_seconds: float = DEFAULT_MIN_WINDOW_AGE
    min_readings_for_features: int = DEFAULT_MIN_READINGS
    max_window_size: int = DEFAULT_MAX_WINDOW_SIZE
    sampling_frequency_hz: float = DEFAULT_SAMPLING_HZ


class SensorFeatureConfigRegistry:
    """Resuelve config por sensor_id > sensor_type > default global."""

    def __init__(
        self,
        by_type_json: Optional[str] = None,
        by_id_json: Optional[str] = None,
        default: Optional[SensorFeatureConfig] = None,
    ) -> None:
        self._default = default or SensorFeatureConfig()
        self._by_type: Dict[str, SensorFeatureConfig] = {}
        self._by_id: Dict[int, SensorFeatureConfig] = {}
        self._load_env(by_type_json, by_id_json)

    def _load_env(self, by_type_json: Optional[str], by_id_json: Optional[str]) -> None:
        raw_type = by_type_json or os.environ.get("ML_FEATURE_CONFIG_BY_TYPE", "{}")
        raw_id = by_id_json or os.environ.get("ML_FEATURE_CONFIG_BY_ID", "{}")
        for raw, dest, label in ((raw_type, self._by_type, "type"), (raw_id, self._by_id, "id")):
            if not raw or raw == "{}":
                continue
            try:
                data = json.loads(raw)
                for key, cfg in data.items():
                    dest[int(key) if label == "id" else key] = SensorFeatureConfig(
                        min_window_age_seconds=cfg.get("min_age_s", self._default.min_window_age_seconds),
                        min_readings_for_features=cfg.get("min_readings", self._default.min_readings_for_features),
                        max_window_size=cfg.get("max_window", self._default.max_window_size),
                        sampling_frequency_hz=cfg.get("sampling_hz", self._default.sampling_frequency_hz),
                    )
                logger.info("[P3-4] Loaded %d configs by %s", len(dest), label)
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning("[P3-4] Invalid JSON for %s config: %s", label, e)

    def get(self, sensor_id: int, sensor_type: Optional[str] = None) -> SensorFeatureConfig:
        if sensor_id in self._by_id:
            return self._by_id[sensor_id]
        if sensor_type and sensor_type in self._by_type:
            return self._by_type[sensor_type]
        return self._default

    def register_sensor(self, sensor_id: int, config: SensorFeatureConfig) -> None:
        self._by_id[sensor_id] = config

    def register_type(self, sensor_type: str, config: SensorFeatureConfig) -> None:
        self._by_type[sensor_type] = config
