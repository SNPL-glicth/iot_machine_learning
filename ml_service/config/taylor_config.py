"""Taylor predictor and Kalman filter configuration.

Extracted from flags.py as part of refactoring Paso 5.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, field_validator

from .parsers import parse_int_set


class TaylorConfig(BaseModel):
    """Configuration for Taylor series prediction and Kalman filtering.
    
    Responsibilities:
    - Taylor polynomial order and horizon
    - Kalman filter process variance and warmup
    - Engine selection logic
    - A/B testing configuration
    """

    # --- Panic Button ---
    ML_ROLLBACK_TO_BASELINE: bool = False

    # --- Core Features ---
    ML_USE_TAYLOR_PREDICTOR: bool = False
    ML_USE_KALMAN_FILTER: bool = False

    # --- Taylor Config ---
    ML_TAYLOR_ORDER: int = 2
    ML_TAYLOR_HORIZON: int = 1

    # --- Kalman Config ---
    ML_KALMAN_Q: float = 1e-5
    ML_KALMAN_WARMUP_SIZE: int = 10

    # --- A/B Testing ---
    ML_ENABLE_AB_TESTING: bool = False

    # --- Sensor Whitelist ---
    ML_TAYLOR_SENSOR_WHITELIST: Optional[str] = None

    # --- Default Engine ---
    ML_DEFAULT_ENGINE: str = "baseline_moving_average"

    # --- Per-series Overrides (agnóstico) ---
    ML_ENGINE_SERIES_OVERRIDES: Dict[str, str] = {}

    # --- Per-sensor Overrides (legacy IoT) ---
    ML_ENGINE_OVERRIDES: Dict[int, str] = {}

    # --- Cache Config ---
    ML_CACHE_TTL_SECONDS: int = 60
    ML_CACHE_MAX_ENTRIES: int = 1000

    @field_validator("ML_TAYLOR_ORDER")
    @classmethod
    def _clamp_taylor_order(cls, v: int) -> int:
        """Clampea orden de Taylor a [1, 3]."""
        return max(1, min(v, 3))

    @field_validator("ML_TAYLOR_HORIZON")
    @classmethod
    def _validate_horizon(cls, v: int) -> int:
        """Horizon debe ser >= 1."""
        return max(1, v)

    @field_validator("ML_KALMAN_WARMUP_SIZE")
    @classmethod
    def _validate_warmup(cls, v: int) -> int:
        """Warmup debe ser >= 2."""
        return max(2, v)

    def is_series_in_whitelist(self, series_id: str) -> bool:
        """Verifica si una serie está en la whitelist de Taylor."""
        if not self.ML_TAYLOR_SENSOR_WHITELIST:
            return True

        allowed = parse_int_set(self.ML_TAYLOR_SENSOR_WHITELIST)
        from iot_machine_learning.domain.validators.input_guard import (
            safe_series_id_to_int,
        )

        sid = safe_series_id_to_int(series_id, fallback=-1)
        if sid == -1:
            return False
        return sid in allowed

    def is_sensor_in_whitelist(self, sensor_id: int) -> bool:
        """Legacy: verifica si un sensor está en la whitelist."""
        return self.is_series_in_whitelist(str(sensor_id))

    def get_active_engine_for_series(self, series_id: str) -> str:
        """Determina qué motor debe usar una serie (agnóstico)."""
        if self.ML_ROLLBACK_TO_BASELINE:
            return "baseline_moving_average"

        # Agnostic overrides first
        if series_id in self.ML_ENGINE_SERIES_OVERRIDES:
            return self.ML_ENGINE_SERIES_OVERRIDES[series_id]

        # Legacy int overrides
        from iot_machine_learning.domain.validators.input_guard import (
            safe_series_id_to_int,
        )

        sid = safe_series_id_to_int(series_id, fallback=-1)
        if sid != -1 and sid in self.ML_ENGINE_OVERRIDES:
            return self.ML_ENGINE_OVERRIDES[sid]

        if self.ML_USE_TAYLOR_PREDICTOR and self.is_series_in_whitelist(series_id):
            return "taylor"

        return self.ML_DEFAULT_ENGINE

    def get_active_engine_name(self, sensor_id: int) -> str:
        """Legacy: determina motor por sensor_id numérico."""
        return self.get_active_engine_for_series(str(sensor_id))
