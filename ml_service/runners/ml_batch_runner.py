"""ML Batch Runner — runner INTERNO del servicio ML (sklearn regression).

⚠️  NO es el orquestador de producción.
    El batch runner de producción vive en:
    ``iot_ingest_services/jobs/ml_batch_runner.py``

Este módulo es un runner secundario que usa sklearn regression +
IsolationForest para predicción y anomalía. Se usa para desarrollo
y testing del stack ML, NO para el loop de producción.

El bridge enterprise (feature flags, adapters, wiring) está conectado
al runner de producción en iot_ingest_services, no aquí.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from sqlalchemy.engine import Connection

# Imports de infraestructura compartida (BD)
from iot_ingest_services.common.db import get_engine

# Imports internos de ML
from iot_machine_learning.ml_service.config.ml_config import GlobalMLConfig
from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags, get_feature_flags
from iot_machine_learning.ml_service.repository.sensor_repository import list_active_sensors
from iot_machine_learning.ml_service.trainers.isolation_trainer import IsolationForestTrainer

# Imports de módulos refactorizados
from .common.sensor_processor import SensorProcessor
from .bridge_config.batch_flags import should_use_enterprise
from .monitoring.ab_metrics import ABMetricsCollector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerConfig:
    """Configuración del runner."""
    interval_seconds: float
    once: bool
    dedupe_minutes: int


def _iter_sensors(conn: Connection) -> Iterable[int]:
    """Itera sobre sensores activos."""
    return list_active_sensors(conn)


class MLBatchRunner:
    """Orquestador de procesamiento batch de sensores.
    
    Coordina:
    - Iteración sobre sensores activos
    - Procesamiento individual via SensorProcessor
    - Routing enterprise/legacy via feature flags
    - Manejo de errores sin romper el batch
    
    Preparado para paralelización futura.
    """
    
    def __init__(
        self,
        ml_cfg: GlobalMLConfig,
        flags: Optional[FeatureFlags] = None,
    ):
        self._ml_cfg = ml_cfg
        self._flags = flags or FeatureFlags()
        self._sensor_processor = SensorProcessor()
        self._iso_trainer = IsolationForestTrainer(ml_cfg.anomaly)
        self._ab_metrics = ABMetricsCollector()
        self._enterprise_container = None
    
    def _get_enterprise_adapter(self, engine):
        """Lazy-init enterprise container and return adapter, or None."""
        if self._enterprise_container is None:
            try:
                from .wiring.container import BatchEnterpriseContainer
                self._enterprise_container = BatchEnterpriseContainer(
                    engine=engine,
                    flags=self._flags,
                )
            except Exception as exc:
                logger.warning(
                    "[ML_BATCH] Enterprise container init failed: %s", exc
                )
                return None
        return self._enterprise_container.get_prediction_adapter()

    def run_once(self) -> int:
        """Ejecuta un ciclo de procesamiento.
        
        Returns:
            Número de sensores procesados exitosamente
        """
        engine = get_engine()
        processed = 0
        errors = 0
        enterprise_count = 0
        baseline_count = 0
        
        # Lazy-init enterprise adapter (None if flags off or init fails)
        enterprise_adapter_instance = None
        any_enterprise = (
            self._flags.ML_BATCH_USE_ENTERPRISE
            or self._flags.ML_BATCH_ENTERPRISE_SENSORS
        )
        if any_enterprise and not self._flags.ML_ROLLBACK_TO_BASELINE:
            try:
                enterprise_adapter_instance = self._get_enterprise_adapter(engine)
            except Exception as exc:
                logger.warning("[ML_BATCH] Enterprise adapter unavailable: %s", exc)
        
        with engine.begin() as conn:
            sensors = list(_iter_sensors(conn))
            total = len(sensors)
            
            logger.info("[ML_BATCH] Iniciando ciclo: %d sensores", total)
            
            for sensor_id in sensors:
                try:
                    # Decide route per sensor
                    use_enterprise = (
                        enterprise_adapter_instance is not None
                        and should_use_enterprise(sensor_id, self._flags)
                    )
                    adapter_for_sensor = (
                        enterprise_adapter_instance if use_enterprise else None
                    )

                    self._sensor_processor.process(
                        conn,
                        sensor_id,
                        self._ml_cfg,
                        self._iso_trainer,
                        enterprise_adapter=adapter_for_sensor,
                    )
                    processed += 1

                    if use_enterprise:
                        enterprise_count += 1
                    else:
                        baseline_count += 1

                except Exception:
                    errors += 1
                    logger.exception("[ML_BATCH] Error procesando sensor_id=%s", sensor_id)
        
        logger.info(
            "[ML_BATCH] Ciclo completado: %d/%d procesados, %d errores, "
            "enterprise=%d baseline=%d",
            processed, total, errors, enterprise_count, baseline_count,
        )
        return processed
    
    def run_loop(self, config: RunnerConfig) -> None:
        """Ejecuta el runner en loop.
        
        Args:
            config: Configuración del runner
        """
        logger.info("[ML_BATCH] Iniciando ML batch runner")
        
        while True:
            logger.info("[ML_BATCH] Inicio iteración")
            self.run_once()
            logger.info("[ML_BATCH] Fin iteración")
            
            if config.once:
                break
            
            time.sleep(config.interval_seconds)


# ---------------------------------------------------------------------------
# Funciones de compatibilidad con código existente
# ---------------------------------------------------------------------------

def run_once(ml_cfg: GlobalMLConfig, dedupe_minutes: int) -> None:
    """Función de compatibilidad con código existente."""
    runner = MLBatchRunner(ml_cfg)
    runner.run_once()


def main() -> None:
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="ML batch runner (sklearn regression + IsolationForest)"
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=60.0,
        help="Intervalo entre ejecuciones (segundos). Ignorado si se usa --once.",
    )
    parser.add_argument(
        "--dedupe-minutes",
        type=int,
        default=10,
        help="Minutos para deduplicar eventos de cruce de umbral.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Ejecutar solo una vez y salir.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    logger.info("[ML_BATCH] Iniciando ML batch runner (sklearn + IsolationForest)")

    ml_cfg = GlobalMLConfig()
    flags = get_feature_flags()
    config = RunnerConfig(
        interval_seconds=args.interval_seconds,
        once=bool(args.once),
        dedupe_minutes=args.dedupe_minutes,
    )

    runner = MLBatchRunner(ml_cfg, flags=flags)
    runner.run_loop(config)


if __name__ == "__main__":
    main()
