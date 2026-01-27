"""ML Batch Runner - ORQUESTADOR PURO.

Este módulo coordina el procesamiento batch de sensores sin implementar
lógica de negocio directamente. Toda la lógica está delegada a:
- common/sensor_processor: Procesamiento individual de sensores
- common/model_manager: Gestión de modelos ML
- common/prediction_writer: Persistencia de predicciones
- common/event_writer: Gestión de eventos ML
- common/severity_classifier: Clasificación de severidad

Complejidad: O(n) donde n = número de sensores activos.
Preparado para paralelización por sensor.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy.engine import Connection

# Imports de infraestructura compartida (BD)
from iot_ingest_services.common.db import get_engine

# Imports internos de ML
from iot_machine_learning.ml_service.config.ml_config import GlobalMLConfig
from iot_machine_learning.ml_service.repository.sensor_repository import list_active_sensors
from iot_machine_learning.ml_service.trainers.isolation_trainer import IsolationForestTrainer

# Imports de módulos refactorizados
from .common.sensor_processor import SensorProcessor

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
    - Manejo de errores sin romper el batch
    
    Preparado para paralelización futura.
    """
    
    def __init__(self, ml_cfg: GlobalMLConfig):
        self._ml_cfg = ml_cfg
        self._sensor_processor = SensorProcessor()
        self._iso_trainer = IsolationForestTrainer(ml_cfg.anomaly)
    
    def run_once(self) -> int:
        """Ejecuta un ciclo de procesamiento.
        
        Returns:
            Número de sensores procesados exitosamente
        """
        engine = get_engine()
        processed = 0
        errors = 0
        
        with engine.begin() as conn:
            sensors = list(_iter_sensors(conn))
            total = len(sensors)
            
            logger.info("[ML_BATCH] Iniciando ciclo: %d sensores", total)
            
            for sensor_id in sensors:
                try:
                    self._sensor_processor.process(
                        conn,
                        sensor_id,
                        self._ml_cfg,
                        self._iso_trainer,
                    )
                    processed += 1
                except Exception:
                    errors += 1
                    logger.exception("[ML_BATCH] Error procesando sensor_id=%s", sensor_id)
        
        logger.info(
            "[ML_BATCH] Ciclo completado: %d/%d procesados, %d errores",
            processed, total, errors
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
    config = RunnerConfig(
        interval_seconds=args.interval_seconds,
        once=bool(args.once),
        dedupe_minutes=args.dedupe_minutes,
    )

    runner = MLBatchRunner(ml_cfg)
    runner.run_loop(config)


if __name__ == "__main__":
    main()
