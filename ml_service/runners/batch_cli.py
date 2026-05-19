"""CLI entry-point para el ML batch runner.

Extraído de ml_batch_runner.py para modularidad (≤180 líneas).
"""

from __future__ import annotations

import argparse
import logging

from .ml_batch_runner import MLBatchRunner, RunnerConfig, run_once
from iot_machine_learning.ml_service.config.ml_config import GlobalMLConfig
from iot_machine_learning.ml_service.config.feature_flags import get_feature_flags

logger = logging.getLogger(__name__)


def main() -> None:
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(description="ML batch runner (sklearn + IsolationForest)")
    parser.add_argument("--interval-seconds", type=float, default=60.0,
                        help="Intervalo entre ejecuciones (segundos). Ignorado con --once.")
    parser.add_argument("--dedupe-minutes", type=int, default=10,
                        help="Minutos para deduplicar eventos de cruce de umbral.")
    parser.add_argument("--once", action="store_true", help="Ejecutar solo una vez y salir.")
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
