"""Worker individual para procesar un sensor en batch runner.

Extraído de ml_batch_runner.py para modularidad (≤180 líneas).
Cada worker abre su propia conexión SQL (FIX P0-1).
"""

from __future__ import annotations

import logging
from typing import Optional

from iot_ingest_services.common.db import get_engine
from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
from iot_machine_learning.ml_service.config.ml_config import GlobalMLConfig
from iot_machine_learning.ml_service.trainers.isolation_trainer import IsolationForestTrainer

from .common.sensor_processor import SensorProcessor
from .bridge_config.batch_flags import should_use_enterprise

logger = logging.getLogger(__name__)


def process_sensor(
    sensor_id: int,
    enterprise_adapter: Optional[object],
    sensor_processor: SensorProcessor,
    ml_cfg: GlobalMLConfig,
    iso_trainer: IsolationForestTrainer,
    flags: FeatureFlags,
) -> dict:
    """Procesa un sensor individual con su propia conexión SQL.

    Retorna dict con keys: sensor_id, ok, enterprise, error.
    """
    result = {"sensor_id": sensor_id, "ok": False, "enterprise": False, "error": None}
    try:
        use_enterprise = enterprise_adapter is not None and should_use_enterprise(sensor_id, flags)
        adapter = enterprise_adapter if use_enterprise else None

        engine = get_engine()
        with engine.begin() as conn:
            sensor_processor.process(
                conn, sensor_id, ml_cfg, iso_trainer, enterprise_adapter=adapter
            )
        result["ok"] = True
        result["enterprise"] = use_enterprise
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "ml_batch_sensor_error",
            extra={"sensor_id": sensor_id, "error": result["error"], "enterprise": result["enterprise"]},
        )
    return result
