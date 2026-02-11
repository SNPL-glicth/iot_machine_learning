"""Caso de uso: selección de motor de predicción por feature flags.

Extraído de ml/core/engine_factory.py.get_engine_for_sensor().
Responsabilidad ÚNICA: dado un sensor_id y feature flags,
decidir qué motor de predicción usar.

Esto es lógica de APLICACIÓN (no de factory ni de dominio):
- Lee configuración (feature flags)
- Aplica reglas de selección (prioridad)
- Delega creación al EngineRegistry
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


def select_engine_for_sensor(
    sensor_id: int,
    flags: "FeatureFlags",
) -> dict:
    """Selecciona motor de predicción según feature flags.

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
