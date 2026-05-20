"""ML Batch Runner — runner INTERNO del servicio ML (sklearn regression)."""
from __future__ import annotations

from dotenv import load_dotenv
import os
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(_env_path)

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Optional

from sqlalchemy.engine import Connection

from iot_ingest_services.common.db import get_engine
from iot_machine_learning.ml_service.config.ml_config import GlobalMLConfig
from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
from iot_machine_learning.ml_service.repository.sensor_repository import list_active_sensors
from iot_machine_learning.ml_service.trainers.isolation_trainer import IsolationForestTrainer

from .batch_worker import process_sensor
from .common.sensor_processor import SensorProcessor
from .bridge_config.batch_flags import should_use_enterprise
from .monitoring.ab_metrics import ABMetricsCollector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerConfig:
    interval_seconds: float
    once: bool
    dedupe_minutes: int


def _iter_sensors(conn: Connection) -> Iterable[int]:
    return list_active_sensors(conn)


class MLBatchRunner:
    """Orquestador de procesamiento batch de sensores."""

    def __init__(self, ml_cfg: GlobalMLConfig, flags: Optional[FeatureFlags] = None):
        self._ml_cfg = ml_cfg
        self._flags = flags or FeatureFlags()
        self._sensor_processor = SensorProcessor()
        self._iso_trainer = IsolationForestTrainer(ml_cfg.anomaly)
        self._ab_metrics = ABMetricsCollector()
        self._enterprise_container = None
        self._cognitive_sensor_ids = self._parse_cognitive_sensor_ids()

    def _parse_cognitive_sensor_ids(self) -> Optional[set]:
        """FIX P2-3: Parse ML_COGNITIVE_SENSOR_IDS. Returns None if '*' (all)."""
        raw = os.getenv("ML_COGNITIVE_SENSOR_IDS", "").strip()
        if not raw:
            return set()  # Ninguno usa cognitive
        if raw == "*":
            return None  # Todos usan cognitive (comportamiento ML_USE_COGNITIVE_ORCHESTRATOR=True)
        try:
            ids = {int(s.strip()) for s in raw.split(",") if s.strip()}
        except ValueError:
            logger.warning("ml_cognitive_sensor_ids_invalid", extra={"raw": raw})
            return set()
        count = len(ids)
        if count <= 20:
            logger.info("cognitive_enabled_for_sensors", extra={"sensor_ids": sorted(ids)})
        else:
            logger.info("cognitive_enabled_for_N_sensors", extra={"count": count})
        return ids

    def _use_cognitive_for(self, sensor_id: int) -> bool:
        """Decide si un sensor usa el cognitive orchestrator."""
        if self._flags.ML_USE_COGNITIVE_ORCHESTRATOR:
            return True  # Override global (retrocompat)
        if self._cognitive_sensor_ids is None:
            return True  # ML_COGNITIVE_SENSOR_IDS=*
        return sensor_id in self._cognitive_sensor_ids

    def _get_enterprise_adapter(self, engine, sensor_id: int = 0):
        if self._enterprise_container is None:
            try:
                from .wiring.container import BatchEnterpriseContainer
                self._enterprise_container = BatchEnterpriseContainer(engine=engine, flags=self._flags)
                logger.info("[ML_BATCH] Enterprise container initialized successfully")
            except Exception as exc:
                logger.exception("[ML_BATCH] Enterprise container init failed: %s", exc)
                return None
        try:
            if self._use_cognitive_for(sensor_id):
                adapter = self._enterprise_container.get_cognitive_adapter()
            else:
                adapter = self._enterprise_container.get_prediction_adapter()
            return adapter
        except Exception as exc:
            logger.exception("[ML_BATCH] Failed to get prediction adapter: %s", exc)
            return None

    def run_once(self) -> int:
        """Ejecuta un ciclo de procesamiento.

        FIX P0-1: Paralelización con ThreadPoolExecutor.
        Si ML_BATCH_MAX_WORKERS=1, comportamiento idéntico al secuencial.
        """
        t_start = time.monotonic()
        engine = get_engine()
        processed = 0
        errors = 0
        enterprise_count = 0
        baseline_count = 0

        enterprise_adapter = None
        any_enterprise = self._flags.ML_BATCH_USE_ENTERPRISE or self._flags.ML_BATCH_ENTERPRISE_SENSORS
        if any_enterprise and not self._flags.ML_ROLLBACK_TO_BASELINE:
            try:
                enterprise_adapter = self._get_enterprise_adapter(engine)
            except Exception as exc:
                logger.exception("[ML_BATCH] Enterprise adapter unavailable: %s", exc)

        with engine.begin() as conn:
            sensors = list(_iter_sensors(conn))
        total = len(sensors)

        max_workers = int(os.getenv("ML_BATCH_MAX_WORKERS", "4"))
        logger.info(
            "ml_batch_cycle_start",
            extra={"total_sensors": total, "max_workers": max_workers,
                   "enterprise_mode": any_enterprise and not self._flags.ML_ROLLBACK_TO_BASELINE},
        )

        if max_workers <= 1:
            for sensor_id in sensors:
                result = process_sensor(
                    sensor_id, enterprise_adapter, self._sensor_processor,
                    self._ml_cfg, self._iso_trainer, self._flags,
                )
                if result["ok"]:
                    processed += 1
                    enterprise_count += 1 if result["enterprise"] else 0
                    baseline_count += 0 if result["enterprise"] else 1
                else:
                    errors += 1
                    logger.exception("[ML_BATCH] Error procesando sensor_id=%s", sensor_id)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        process_sensor, sensor_id, enterprise_adapter,
                        self._sensor_processor, self._ml_cfg, self._iso_trainer, self._flags,
                    ): sensor_id for sensor_id in sensors
                }
                for future in as_completed(futures):
                    sensor_id = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        # PERF-P0: Error isolation — one sensor failure
                        # must not crash the entire batch.
                        errors += 1
                        logger.error(
                            "ml_batch_sensor_exception",
                            extra={
                                "sensor_id": sensor_id,
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                            },
                        )
                        continue
                    if result["ok"]:
                        processed += 1
                        enterprise_count += 1 if result["enterprise"] else 0
                        baseline_count += 0 if result["enterprise"] else 1
                    else:
                        errors += 1

        elapsed = time.monotonic() - t_start
        logger.info(
            "ml_batch_cycle_complete",
            extra={"total_sensors": total, "ok_count": processed, "error_count": errors,
                   "enterprise_count": enterprise_count, "baseline_count": baseline_count,
                   "elapsed_seconds": round(elapsed, 2), "max_workers": max_workers},
        )
        return processed

    def run_loop(self, config: RunnerConfig) -> None:
        logger.info("[ML_BATCH] Iniciando ML batch runner")
        while True:
            self.run_once()
            if config.once:
                break
            time.sleep(config.interval_seconds)

def run_once(ml_cfg: GlobalMLConfig, dedupe_minutes: int) -> None:  # noqa: ARG001
    MLBatchRunner(ml_cfg).run_once()
