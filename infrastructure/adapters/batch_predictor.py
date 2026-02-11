"""Procesamiento batch optimizado con paralelización y circuit breaker.

Procesa múltiples sensores en paralelo usando ThreadPoolExecutor.
Incluye circuit breaker para fallar rápido si muchos sensores fallan.

Performance objetivo:
- 100 sensores en <5 segundos (vs 20-30s secuencial)
- Circuit breaker se activa después de N errores consecutivos

Thread-safe: cada worker opera independientemente.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

from ...domain.entities.prediction import Prediction
from ...domain.entities.sensor_reading import SensorWindow
from ...domain.ports.prediction_port import PredictionPort

logger = logging.getLogger(__name__)


class CircuitBreakerOpen(Exception):
    """Excepción cuando el circuit breaker está abierto."""

    pass


class BatchPredictor:
    """Procesamiento batch de predicciones con paralelización.

    Attributes:
        _max_workers: Threads paralelos.
        _circuit_breaker_threshold: Errores consecutivos para activar CB.
        _timeout_per_sensor_s: Timeout por sensor individual.
        _consecutive_errors: Contador de errores consecutivos.
        _circuit_open: ``True`` si el circuit breaker está abierto.
    """

    def __init__(
        self,
        max_workers: int = 4,
        circuit_breaker_threshold: int = 10,
        timeout_per_sensor_s: float = 5.0,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers debe ser >= 1, recibido {max_workers}")

        self._max_workers = max_workers
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._timeout_per_sensor_s = timeout_per_sensor_s
        self._consecutive_errors: int = 0
        self._circuit_open: bool = False

    def predict_batch(
        self,
        sensor_ids: List[int],
        engine: PredictionPort,
        data_loader: Callable[[int], SensorWindow],
    ) -> Dict[int, Prediction]:
        """Predice para múltiples sensores en paralelo.

        Args:
            sensor_ids: IDs de sensores a procesar.
            engine: Motor de predicción a usar.
            data_loader: Función que carga datos: ``sensor_id → SensorWindow``.

        Returns:
            Dict ``{sensor_id: Prediction}``.  Sensores que fallaron se omiten.

        Raises:
            CircuitBreakerOpen: Si el circuit breaker está activo.
        """
        if self._circuit_open:
            raise CircuitBreakerOpen(
                f"Circuit breaker abierto después de "
                f"{self._circuit_breaker_threshold} errores consecutivos"
            )

        t_start = time.monotonic()
        results: Dict[int, Prediction] = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(
                    self._predict_single, sid, engine, data_loader
                ): sid
                for sid in sensor_ids
            }

            for future in as_completed(futures):
                sensor_id = futures[future]

                try:
                    result = future.result(timeout=self._timeout_per_sensor_s)
                    if result is not None:
                        results[sensor_id] = result
                        self._consecutive_errors = 0  # Reset on success

                except Exception as exc:
                    self._consecutive_errors += 1
                    logger.error(
                        "batch_sensor_failed",
                        extra={
                            "sensor_id": sensor_id,
                            "error": str(exc),
                            "consecutive_errors": self._consecutive_errors,
                        },
                    )

                    # Circuit breaker
                    if self._consecutive_errors >= self._circuit_breaker_threshold:
                        self._circuit_open = True
                        logger.critical(
                            "circuit_breaker_activated",
                            extra={
                                "threshold": self._circuit_breaker_threshold,
                                "processed": len(results),
                                "total": len(sensor_ids),
                            },
                        )
                        break

        elapsed_s = time.monotonic() - t_start

        logger.info(
            "batch_prediction_complete",
            extra={
                "total_sensors": len(sensor_ids),
                "successful": len(results),
                "failed": len(sensor_ids) - len(results),
                "elapsed_s": round(elapsed_s, 3),
                "avg_ms_per_sensor": round(
                    (elapsed_s * 1000) / max(len(sensor_ids), 1), 1
                ),
            },
        )

        return results

    def reset_circuit_breaker(self) -> None:
        """Resetea el circuit breaker manualmente."""
        self._circuit_open = False
        self._consecutive_errors = 0
        logger.info("circuit_breaker_reset")

    @property
    def is_circuit_open(self) -> bool:
        """``True`` si el circuit breaker está activo."""
        return self._circuit_open

    def _predict_single(
        self,
        sensor_id: int,
        engine: PredictionPort,
        data_loader: Callable[[int], SensorWindow],
    ) -> Optional[Prediction]:
        """Predice para un sensor individual (worker thread).

        Args:
            sensor_id: ID del sensor.
            engine: Motor de predicción.
            data_loader: Función de carga de datos.

        Returns:
            ``Prediction`` o ``None`` si falla.
        """
        try:
            window = data_loader(sensor_id)
            if window.is_empty:
                logger.debug(
                    "batch_sensor_no_data",
                    extra={"sensor_id": sensor_id},
                )
                return None

            return engine.predict(window)

        except Exception as exc:
            logger.warning(
                "batch_single_prediction_failed",
                extra={"sensor_id": sensor_id, "error": str(exc)},
            )
            raise
