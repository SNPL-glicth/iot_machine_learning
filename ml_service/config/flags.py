"""Feature flags model for UTSAE control.

Todos los flags tienen valores por defecto SEGUROS (desactivados).
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, field_validator

from .parsers import parse_int_set


class FeatureFlags(BaseModel):
    """Feature flags para control de features UTSAE.

    Todos los valores por defecto son SEGUROS: el sistema se comporta
    exactamente como antes de UTSAE si no se configura nada.

    Attributes:
        ML_ROLLBACK_TO_BASELINE: Panic button.  Si ``True``, TODO usa
            baseline sin importar otros flags.
        ML_USE_TAYLOR_PREDICTOR: Activa el motor Taylor.
        ML_USE_KALMAN_FILTER: Activa el filtro Kalman pre-predicción.
        ML_TAYLOR_ORDER: Orden de Taylor (1–3).
        ML_TAYLOR_HORIZON: Pasos adelante a predecir.
        ML_KALMAN_Q: Varianza del proceso para Kalman.
        ML_KALMAN_WARMUP_SIZE: Lecturas de warmup para Kalman.
        ML_ENABLE_AB_TESTING: Activa comparación A/B baseline vs Taylor.
        ML_TAYLOR_SENSOR_WHITELIST: CSV de sensor_ids para Taylor.
            Si vacío/None, Taylor aplica a todos (si está activado).
        ML_DEFAULT_ENGINE: Motor por defecto.
        ML_ENGINE_OVERRIDES: Overrides por sensor_id.
    """

    # --- PANIC BUTTON ---
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

    # --- Enterprise Features (Fase 3) ---
    ML_ENABLE_DELTA_SPIKE_DETECTION: bool = False
    ML_ENABLE_REGIME_DETECTION: bool = False
    ML_ENABLE_ENSEMBLE_PREDICTOR: bool = False
    ML_ENABLE_AUDIT_LOGGING: bool = False
    ML_ENABLE_PREDICTION_CACHE: bool = False
    ML_ENABLE_VOTING_ANOMALY: bool = False
    ML_ENABLE_CHANGE_POINT_DETECTION: bool = False
    ML_ENABLE_EXPLAINABILITY: bool = False

    # --- Cache Config ---
    ML_CACHE_TTL_SECONDS: int = 60
    ML_CACHE_MAX_ENTRIES: int = 1000

    # --- Batch Config ---
    ML_BATCH_MAX_WORKERS: int = 4
    ML_BATCH_CIRCUIT_BREAKER_THRESHOLD: int = 10

    # --- Anomaly Config ---
    ML_ANOMALY_VOTING_THRESHOLD: float = 0.5
    ML_ANOMALY_CONTAMINATION: float = 0.1

    # --- Batch Runner Enterprise Bridge ---
    ML_BATCH_USE_ENTERPRISE: bool = False
    ML_BATCH_ENTERPRISE_SENSORS: Optional[str] = None
    ML_BATCH_BASELINE_ONLY_SENSORS: Optional[str] = None

    # --- Performance Fixes (H-ML-4, H-ML-8, H-ING-2) ---
    ML_ENTERPRISE_USE_PRELOADED_DATA: bool = True
    ML_STREAM_USE_SLIDING_WINDOW: bool = True
    ML_MQTT_ASYNC_PROCESSING: bool = True
    ML_MQTT_QUEUE_SIZE: int = 1000
    ML_MQTT_NUM_WORKERS: int = 4

    # --- Batch Parallelism (E-4 / RC-2 fix) ---
    ML_BATCH_PARALLEL_WORKERS: int = 1  # 1 = sequential (backward compat)

    # --- Stream Prediction Dedup (E-2 / RC-1 fix) ---
    ML_STREAM_PREDICTIONS_ENABLED: bool = False  # Default: batch only

    # --- Sliding Window Eviction (E-1 / RC-3 fix) ---
    ML_SLIDING_WINDOW_MAX_SENSORS: int = 1000
    ML_SLIDING_WINDOW_TTL_SECONDS: int = 3600

    # --- Ingest Circuit Breaker (E-3 / RC-4 fix) ---
    ML_INGEST_CIRCUIT_BREAKER_ENABLED: bool = True
    ML_INGEST_CB_FAILURE_THRESHOLD: int = 5
    ML_INGEST_CB_TIMEOUT_SECONDS: int = 30

    # --- Cognitive Memory (Weaviate) ---
    ML_ENABLE_COGNITIVE_MEMORY: bool = False
    ML_COGNITIVE_MEMORY_DRY_RUN: bool = True
    ML_COGNITIVE_MEMORY_ASYNC: bool = True
    ML_COGNITIVE_MEMORY_URL: str = ""
    ML_ENABLE_MEMORY_RECALL: bool = False

    # --- Multi-Head Attention (Document Context) ---
    ML_ENABLE_ATTENTION: bool = False  # Master switch (default: disabled)
    ML_ATTENTION_CONFIDENCE_THRESHOLD: float = 0.5  # Fallback threshold
    ML_ATTENTION_BUDGET_MS: float = 100.0  # Time budget per document

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
        """Verifica si una serie está en la whitelist de Taylor.

        Si la whitelist está vacía o es ``None``, TODAS las series
        están permitidas (whitelist abierta).

        Args:
            series_id: Identificador de la serie.

        Returns:
            ``True`` si la serie puede usar Taylor.
        """
        if not self.ML_TAYLOR_SENSOR_WHITELIST:
            return True

        allowed = parse_int_set(self.ML_TAYLOR_SENSOR_WHITELIST)
        from iot_machine_learning.domain.validators.input_guard import safe_series_id_to_int
        sid = safe_series_id_to_int(series_id, fallback=-1)
        if sid == -1:
            return False
        return sid in allowed

    def is_sensor_in_whitelist(self, sensor_id: int) -> bool:
        """Legacy: verifica si un sensor está en la whitelist."""
        return self.is_series_in_whitelist(str(sensor_id))

    def get_active_engine_for_series(self, series_id: str) -> str:
        """Determina qué motor debe usar una serie (agnóstico).

        Prioridad:
        1. Panic button → baseline
        2. Override por series_id (agnóstico)
        3. Override por sensor_id (legacy, si series_id es numérico)
        4. Taylor (si activo y serie en whitelist)
        5. Default global

        Args:
            series_id: Identificador de la serie.

        Returns:
            Nombre del motor a usar.
        """
        if self.ML_ROLLBACK_TO_BASELINE:
            return "baseline_moving_average"

        # Agnostic overrides first
        if series_id in self.ML_ENGINE_SERIES_OVERRIDES:
            return self.ML_ENGINE_SERIES_OVERRIDES[series_id]

        # Legacy int overrides
        from iot_machine_learning.domain.validators.input_guard import safe_series_id_to_int
        sid = safe_series_id_to_int(series_id, fallback=-1)
        if sid != -1 and sid in self.ML_ENGINE_OVERRIDES:
            return self.ML_ENGINE_OVERRIDES[sid]

        if self.ML_USE_TAYLOR_PREDICTOR and self.is_series_in_whitelist(series_id):
            return "taylor"

        return self.ML_DEFAULT_ENGINE

    # --- Decision Engine (NEW) ---
    ML_ENABLE_DECISION_ENGINE: bool = False  # Default: disabled (safe)
    ML_DECISION_ENGINE_STRATEGY: str = "simple"  # simple | conservative | aggressive | cost_optimized

    @field_validator("ML_DECISION_ENGINE_STRATEGY")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        """Valida que la estrategia sea conocida."""
        allowed = {"simple", "conservative", "aggressive", "cost_optimized"}
        if v not in allowed:
            return "simple"  # Fallback seguro
        return v

    def get_active_engine_name(self, sensor_id: int) -> str:
        """Legacy: determina motor por sensor_id numérico."""
        return self.get_active_engine_for_series(str(sensor_id))
