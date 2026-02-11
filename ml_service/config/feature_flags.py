"""Feature flags para control de features UTSAE.

Todos los flags tienen valores por defecto SEGUROS (desactivados).
Esto garantiza que el sistema funciona igual que antes si no se
configuran variables de entorno.

Carga desde variables de entorno con prefijo ``ML_``.
Ejemplo:
    export ML_USE_TAYLOR_PREDICTOR=true
    export ML_TAYLOR_ORDER=2
    export ML_TAYLOR_SENSOR_WHITELIST=1,5,42

Decisiones de diseño:
- Pydantic BaseModel para validación automática de tipos.
- ``from_env()`` como factory method (no en __init__) para separar
  la lógica de parsing de env vars de la construcción del objeto.
- ``ML_ROLLBACK_TO_BASELINE`` es el "panic button": si está activo,
  TODO el sistema usa baseline sin importar otros flags.
- Whitelist de sensores como string CSV para facilitar configuración
  vía env vars (los dicts no son prácticos en env vars).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Set

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


def _parse_bool(value: str) -> bool:
    """Parsea string a bool (case-insensitive).

    Args:
        value: String a parsear.

    Returns:
        ``True`` para "true", "1", "yes", "on".
        ``False`` para todo lo demás.
    """
    return value.strip().lower() in ("true", "1", "yes", "on")


def _parse_int_set(value: Optional[str]) -> Set[int]:
    """Parsea string CSV de enteros a set.

    Args:
        value: String como ``"1,5,42"`` o ``None``.

    Returns:
        Set de enteros.  Vacío si ``value`` es ``None`` o vacío.
    """
    if not value or not value.strip():
        return set()

    result: Set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if part:
            try:
                result.add(int(part))
            except ValueError:
                logger.warning(
                    "feature_flag_parse_error",
                    extra={"field": "sensor_whitelist", "invalid_value": part},
                )
    return result


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

    # --- Per-sensor Overrides ---
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

    def is_sensor_in_whitelist(self, sensor_id: int) -> bool:
        """Verifica si un sensor está en la whitelist de Taylor.

        Si la whitelist está vacía o es ``None``, TODOS los sensores
        están permitidos (whitelist abierta).

        Args:
            sensor_id: ID del sensor a verificar.

        Returns:
            ``True`` si el sensor puede usar Taylor.
        """
        if not self.ML_TAYLOR_SENSOR_WHITELIST:
            # Whitelist vacía = todos permitidos
            return True

        allowed = _parse_int_set(self.ML_TAYLOR_SENSOR_WHITELIST)
        return sensor_id in allowed

    def get_active_engine_name(self, sensor_id: int) -> str:
        """Determina qué motor debe usar un sensor.

        Prioridad:
        1. Panic button → baseline
        2. Override por sensor
        3. Taylor (si activo y sensor en whitelist)
        4. Default global

        Args:
            sensor_id: ID del sensor.

        Returns:
            Nombre del motor a usar.
        """
        if self.ML_ROLLBACK_TO_BASELINE:
            return "baseline_moving_average"

        if sensor_id in self.ML_ENGINE_OVERRIDES:
            return self.ML_ENGINE_OVERRIDES[sensor_id]

        if self.ML_USE_TAYLOR_PREDICTOR and self.is_sensor_in_whitelist(sensor_id):
            return "taylor"

        return self.ML_DEFAULT_ENGINE

    @classmethod
    def from_env(cls) -> FeatureFlags:
        """Carga feature flags desde variables de entorno.

        Variables reconocidas (todas opcionales):
        - ``ML_ROLLBACK_TO_BASELINE``: "true"/"false"
        - ``ML_USE_TAYLOR_PREDICTOR``: "true"/"false"
        - ``ML_USE_KALMAN_FILTER``: "true"/"false"
        - ``ML_TAYLOR_ORDER``: "1"/"2"/"3"
        - ``ML_TAYLOR_HORIZON``: "1"+ 
        - ``ML_KALMAN_Q``: float como string
        - ``ML_KALMAN_WARMUP_SIZE``: int como string
        - ``ML_ENABLE_AB_TESTING``: "true"/"false"
        - ``ML_TAYLOR_SENSOR_WHITELIST``: "1,5,42"
        - ``ML_DEFAULT_ENGINE``: nombre del motor

        Returns:
            ``FeatureFlags`` con valores de env vars o defaults seguros.
        """
        kwargs: dict = {}

        # Booleans
        for key in (
            "ML_ROLLBACK_TO_BASELINE",
            "ML_USE_TAYLOR_PREDICTOR",
            "ML_USE_KALMAN_FILTER",
            "ML_ENABLE_AB_TESTING",
        ):
            env_val = os.environ.get(key)
            if env_val is not None:
                kwargs[key] = _parse_bool(env_val)

        # Integers
        for key in ("ML_TAYLOR_ORDER", "ML_TAYLOR_HORIZON", "ML_KALMAN_WARMUP_SIZE"):
            env_val = os.environ.get(key)
            if env_val is not None:
                try:
                    kwargs[key] = int(env_val.strip())
                except ValueError:
                    logger.warning(
                        "feature_flag_parse_error",
                        extra={"field": key, "invalid_value": env_val},
                    )

        # Floats
        for key in ("ML_KALMAN_Q",):
            env_val = os.environ.get(key)
            if env_val is not None:
                try:
                    kwargs[key] = float(env_val.strip())
                except ValueError:
                    logger.warning(
                        "feature_flag_parse_error",
                        extra={"field": key, "invalid_value": env_val},
                    )

        # Strings
        for key in ("ML_TAYLOR_SENSOR_WHITELIST", "ML_DEFAULT_ENGINE"):
            env_val = os.environ.get(key)
            if env_val is not None:
                kwargs[key] = env_val.strip()

        flags = cls(**kwargs)

        logger.info(
            "feature_flags_loaded",
            extra={
                "source": "env",
                "rollback": flags.ML_ROLLBACK_TO_BASELINE,
                "taylor": flags.ML_USE_TAYLOR_PREDICTOR,
                "kalman": flags.ML_USE_KALMAN_FILTER,
                "ab_testing": flags.ML_ENABLE_AB_TESTING,
                "default_engine": flags.ML_DEFAULT_ENGINE,
            },
        )

        return flags


# --- Singleton global (lazy) ---
_global_flags: Optional[FeatureFlags] = None


def get_feature_flags() -> FeatureFlags:
    """Retorna la instancia global de feature flags (lazy singleton).

    La primera llamada carga desde env vars.  Llamadas subsiguientes
    retornan la misma instancia.

    Returns:
        ``FeatureFlags`` global.
    """
    global _global_flags
    if _global_flags is None:
        _global_flags = FeatureFlags.from_env()
    return _global_flags


def reset_feature_flags() -> None:
    """Resetea el singleton (para testing)."""
    global _global_flags
    _global_flags = None
