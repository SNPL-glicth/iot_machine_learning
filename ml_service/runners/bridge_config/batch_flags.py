"""Feature flags para migración gradual del batch runner a enterprise stack.

Niveles de control (prioridad descendente):
1. Panic button global (ML_ROLLBACK_TO_BASELINE) — fuerza baseline
2. Blacklist por sensor (ML_BATCH_BASELINE_ONLY_SENSORS) — fuerza baseline
3. Whitelist por sensor (ML_BATCH_ENTERPRISE_SENSORS) — fuerza enterprise
4. Flag global (ML_BATCH_USE_ENTERPRISE) — enterprise/baseline

Uso típico:
    # Piloto con 3 sensores
    export ML_BATCH_ENTERPRISE_SENSORS="42,55,78"

    # Rollout global
    export ML_BATCH_USE_ENTERPRISE=true

    # Emergencia
    export ML_ROLLBACK_TO_BASELINE=true
"""

from __future__ import annotations

import logging
from typing import Set

from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


def should_use_enterprise(sensor_id: int, flags: FeatureFlags) -> bool:
    """Decide si usar enterprise stack para este sensor.

    Prioridad (de mayor a menor):
    1. Panic button → baseline
    2. Sensor en blacklist → baseline
    3. Sensor en whitelist → enterprise
    4. Flag global → enterprise/baseline
    5. Default → baseline (conservador)

    Args:
        sensor_id: ID del sensor.
        flags: Feature flags globales.

    Returns:
        ``True`` si usar enterprise, ``False`` si baseline.
    """
    # 1. Panic button
    if flags.ML_ROLLBACK_TO_BASELINE:
        return False

    # 2. Blacklist (sensores problemáticos)
    blacklist = _parse_sensor_set(
        getattr(flags, "ML_BATCH_BASELINE_ONLY_SENSORS", None)
    )
    if sensor_id in blacklist:
        return False

    # 3. Whitelist (pilotos)
    whitelist = _parse_sensor_set(
        getattr(flags, "ML_BATCH_ENTERPRISE_SENSORS", None)
    )
    if whitelist and sensor_id in whitelist:
        return True

    # 4. Flag global
    if getattr(flags, "ML_BATCH_USE_ENTERPRISE", False):
        return True

    # 5. Default: baseline (conservador)
    return False


def _parse_sensor_set(csv_string: object) -> Set[int]:
    """Parsea string ``"1,5,42"`` a ``{1, 5, 42}``.

    Returns:
        Set vacío si string es ``None``/vacío.
    """
    if not csv_string:
        return set()

    raw = str(csv_string)
    result: Set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                result.add(int(part))
            except ValueError:
                logger.warning(
                    "batch_flags_parse_error",
                    extra={"invalid_value": part, "source": "sensor_list"},
                )
    return result
