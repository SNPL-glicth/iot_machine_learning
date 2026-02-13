"""Caso de uso: selección de motor de predicción.

Dual interface:
- Legacy (IoT): ``select_engine_for_sensor(sensor_id, flags)`` — por sensor_id + whitelist.
- Agnóstico: ``select_engine_for_series(profile, flags)`` — por características del dato.

Esto es lógica de APLICACIÓN (no de factory ni de dominio):
- Lee configuración (feature flags)
- Aplica reglas de selección (prioridad)
- Delega creación al EngineRegistry
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iot_machine_learning.domain.entities.series_profile import SeriesProfile
    from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


def select_engine_for_sensor(
    sensor_id: int,
    flags: "FeatureFlags",
) -> dict:
    """Selecciona motor de predicción según feature flags.

    .. deprecated::
        Usar ``select_engine_for_series(profile, flags)`` en su lugar.
        Selección por ``sensor_id`` acopla a IoT; selección por
        ``SeriesProfile`` es agnóstica al dominio.

    Retorna un dict con ``{"engine_name": str, "kwargs": dict}``
    que el caller puede usar para instanciar el motor.

    Prioridad de selección:
    1. ``ML_ROLLBACK_TO_BASELINE`` = True → baseline (panic button)
    2. Override por sensor en ``ML_ENGINE_OVERRIDES``
    3. Sensor en whitelist de Taylor → taylor
    4. ``ML_DEFAULT_ENGINE`` global
    5. Fallback a baseline

    Args:
        sensor_id: ID del sensor.
        flags: Instancia de ``FeatureFlags``.

    Returns:
        Dict con ``engine_name`` y ``kwargs`` para creación.
    """
    warnings.warn(
        "select_engine_for_sensor(sensor_id) is deprecated. "
        "Use select_engine_for_series(profile, flags) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # 1. Panic button
    if flags.ML_ROLLBACK_TO_BASELINE:
        logger.info(
            "panic_button_active",
            extra={"sensor_id": sensor_id},
        )
        return {"engine_name": "baseline_moving_average", "kwargs": {}}

    # 2. Override por sensor
    if sensor_id in flags.ML_ENGINE_OVERRIDES:
        engine_name = flags.ML_ENGINE_OVERRIDES[sensor_id]
        return {
            "engine_name": engine_name,
            "kwargs": _kwargs_for_engine(engine_name, flags),
        }

    # 3. Sensor en whitelist de Taylor
    if flags.ML_USE_TAYLOR_PREDICTOR and flags.is_sensor_in_whitelist(sensor_id):
        return {
            "engine_name": "taylor",
            "kwargs": _kwargs_for_engine("taylor", flags),
        }

    # 4. Default global
    default_name = flags.ML_DEFAULT_ENGINE
    return {
        "engine_name": default_name,
        "kwargs": _kwargs_for_engine(default_name, flags),
    }


def _kwargs_for_engine(name: str, flags: "FeatureFlags") -> dict:
    """Genera kwargs de creación según el motor y flags.

    Args:
        name: Nombre del motor.
        flags: FeatureFlags con parámetros.

    Returns:
        Dict de kwargs para el constructor del motor.
    """
    if name == "taylor":
        return {
            "order": flags.ML_TAYLOR_ORDER,
            "horizon": flags.ML_TAYLOR_HORIZON,
        }
    return {}


# ---------------------------------------------------------------------------
# Selección agnóstica por SeriesProfile (Nivel 1 UTSAE)
# ---------------------------------------------------------------------------


def select_engine_for_series(
    profile: "SeriesProfile",
    flags: "FeatureFlags",
) -> dict:
    """Selecciona motor de predicción por características del dato.

    No usa ``sensor_id`` ni whitelists.  Decide por:
    - Estacionariedad (trend → Taylor, stationary → baseline)
    - Volatilidad (high → ensemble)
    - Cantidad de puntos (pocos → baseline)

    Args:
        profile: Perfil estadístico auto-detectado de la serie.
        flags: Feature flags para panic button y parámetros.

    Returns:
        Dict con ``engine_name`` y ``kwargs`` para creación.
    """
    from iot_machine_learning.domain.entities.series_profile import (
        StationarityHint,
        VolatilityLevel,
    )

    # 1. Panic button
    if flags.ML_ROLLBACK_TO_BASELINE:
        return {"engine_name": "baseline_moving_average", "kwargs": {}}

    # 2. Datos insuficientes → baseline
    if not profile.has_sufficient_data:
        return {"engine_name": "baseline_moving_average", "kwargs": {}}

    # 3. Alta volatilidad → ensemble (si disponible)
    if profile.volatility == VolatilityLevel.HIGH:
        return {
            "engine_name": "ensemble_weighted",
            "kwargs": {},
        }

    # 4. Tendencia clara → Taylor
    if profile.stationarity == StationarityHint.TREND and profile.n_points >= 5:
        return {
            "engine_name": "taylor",
            "kwargs": _kwargs_for_engine("taylor", flags),
        }

    # 5. Default global
    return {
        "engine_name": flags.ML_DEFAULT_ENGINE,
        "kwargs": _kwargs_for_engine(flags.ML_DEFAULT_ENGINE, flags),
    }
