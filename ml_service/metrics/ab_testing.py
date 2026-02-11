"""Framework de A/B testing para comparar motores de predicción.

Compara baseline vs Taylor (u otros motores) en paralelo sin afectar
la predicción que se persiste en BD.  Solo registra métricas en memoria
para análisis posterior.

Decisiones de diseño:
- Singleton pattern: una sola instancia acumula datos de todos los
  sensores durante el ciclo de vida del proceso.
- Datos en memoria (no BD): evita modificar schemas.  Se pierde al
  reiniciar, pero el A/B testing es un proceso de evaluación temporal.
- MAE como métrica principal: simple, interpretable, misma unidad que
  los valores del sensor.
- RMSE como métrica secundaria: penaliza errores grandes.
- Winner con margen del 5%: evita declarar ganador por diferencias
  estadísticamente insignificantes.
- Thread-safety con lock: batch y stream pueden registrar en paralelo.
- Límite de muestras por sensor (max_samples): evita crecimiento
  ilimitado de memoria.  Se descartan las más antiguas (FIFO).
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# Mínimo de muestras para calcular resultados significativos
_MIN_SAMPLES: int = 10

# Margen para declarar ganador (5%)
_WINNER_MARGIN: float = 0.05

# Máximo de muestras por sensor (FIFO)
_MAX_SAMPLES_PER_SENSOR: int = 1000


@dataclass(frozen=True)
class ABTestResult:
    """Resultado de comparación A/B para un sensor.

    Attributes:
        sensor_id: ID del sensor evaluado.
        baseline_mae: Mean Absolute Error del baseline.
        baseline_rmse: Root Mean Squared Error del baseline.
        taylor_mae: Mean Absolute Error de Taylor.
        taylor_rmse: Root Mean Squared Error de Taylor.
        winner: Motor ganador (``"baseline"``, ``"taylor"``, ``"tie"``).
        confidence: Confianza basada en número de muestras (0–1).
        n_samples: Número de muestras usadas.
        improvement_pct: Porcentaje de mejora de Taylor sobre baseline.
            Positivo = Taylor mejor.  Negativo = Taylor peor.
    """

    sensor_id: int
    baseline_mae: float
    baseline_rmse: float
    taylor_mae: float
    taylor_rmse: float
    winner: str
    confidence: float
    n_samples: int
    improvement_pct: float


@dataclass
class _SensorData:
    """Datos crudos de A/B testing para un sensor."""

    actual: Deque[float] = field(
        default_factory=lambda: deque(maxlen=_MAX_SAMPLES_PER_SENSOR)
    )
    baseline: Deque[float] = field(
        default_factory=lambda: deque(maxlen=_MAX_SAMPLES_PER_SENSOR)
    )
    taylor: Deque[float] = field(
        default_factory=lambda: deque(maxlen=_MAX_SAMPLES_PER_SENSOR)
    )


class ABTester:
    """Compara baseline vs Taylor en paralelo sin afectar producción.

    Uso típico:
        >>> tester = get_ab_tester()
        >>> tester.record_prediction(sensor_id=1, actual=20.0,
        ...     baseline_pred=20.1, taylor_pred=19.9)
        >>> result = tester.compute_results(sensor_id=1)
        >>> result.winner  # "taylor" | "baseline" | "tie"

    Thread-safe: puede usarse desde batch y stream runners.
    """

    def __init__(self) -> None:
        """Inicializa el A/B tester."""
        self._raw_data: Dict[int, _SensorData] = {}
        self._results: Dict[int, ABTestResult] = {}
        self._lock: threading.Lock = threading.Lock()

    def record_prediction(
        self,
        sensor_id: int,
        actual_value: float,
        baseline_pred: float,
        taylor_pred: float,
    ) -> None:
        """Registra una predicción de ambos engines para comparación.

        Args:
            sensor_id: ID del sensor.
            actual_value: Valor real observado.
            baseline_pred: Predicción del baseline.
            taylor_pred: Predicción de Taylor.
        """
        # Guard: no registrar NaN/Inf
        for val, name in [
            (actual_value, "actual"),
            (baseline_pred, "baseline"),
            (taylor_pred, "taylor"),
        ]:
            if not math.isfinite(val):
                logger.warning(
                    "ab_test_invalid_value",
                    extra={"sensor_id": sensor_id, "field": name, "value": val},
                )
                return

        with self._lock:
            if sensor_id not in self._raw_data:
                self._raw_data[sensor_id] = _SensorData()

            data = self._raw_data[sensor_id]
            data.actual.append(actual_value)
            data.baseline.append(baseline_pred)
            data.taylor.append(taylor_pred)

        logger.debug(
            "ab_test_recorded",
            extra={
                "sensor_id": sensor_id,
                "actual": actual_value,
                "baseline_pred": baseline_pred,
                "taylor_pred": taylor_pred,
            },
        )

    def compute_results(self, sensor_id: int) -> Optional[ABTestResult]:
        """Calcula MAE/RMSE y determina ganador para un sensor.

        Requiere al menos ``_MIN_SAMPLES`` muestras.

        Args:
            sensor_id: ID del sensor.

        Returns:
            ``ABTestResult`` o ``None`` si no hay suficientes muestras.
        """
        with self._lock:
            data = self._raw_data.get(sensor_id)
            if data is None or len(data.actual) < _MIN_SAMPLES:
                return None

            # Copiar datos bajo lock para calcular fuera del lock
            actual = list(data.actual)
            baseline_preds = list(data.baseline)
            taylor_preds = list(data.taylor)

        n = len(actual)

        # MAE
        baseline_errors = [abs(a - b) for a, b in zip(actual, baseline_preds)]
        taylor_errors = [abs(a - t) for a, t in zip(actual, taylor_preds)]

        baseline_mae = sum(baseline_errors) / n
        taylor_mae = sum(taylor_errors) / n

        # RMSE
        baseline_sq_errors = [e * e for e in baseline_errors]
        taylor_sq_errors = [e * e for e in taylor_errors]

        baseline_rmse = math.sqrt(sum(baseline_sq_errors) / n)
        taylor_rmse = math.sqrt(sum(taylor_sq_errors) / n)

        # Winner con margen
        if baseline_mae > 0 and taylor_mae < baseline_mae * (1.0 - _WINNER_MARGIN):
            winner = "taylor"
        elif taylor_mae > 0 and baseline_mae < taylor_mae * (1.0 - _WINNER_MARGIN):
            winner = "baseline"
        else:
            winner = "tie"

        # Improvement %
        if baseline_mae > 1e-12:
            improvement_pct = ((baseline_mae - taylor_mae) / baseline_mae) * 100.0
        else:
            improvement_pct = 0.0

        # Confianza basada en número de muestras (satura en 100)
        confidence = min(1.0, n / 100.0)

        result = ABTestResult(
            sensor_id=sensor_id,
            baseline_mae=baseline_mae,
            baseline_rmse=baseline_rmse,
            taylor_mae=taylor_mae,
            taylor_rmse=taylor_rmse,
            winner=winner,
            confidence=confidence,
            n_samples=n,
            improvement_pct=improvement_pct,
        )

        with self._lock:
            self._results[sensor_id] = result

        logger.info(
            "ab_test_result",
            extra={
                "sensor_id": sensor_id,
                "winner": winner,
                "baseline_mae": round(baseline_mae, 6),
                "taylor_mae": round(taylor_mae, 6),
                "baseline_rmse": round(baseline_rmse, 6),
                "taylor_rmse": round(taylor_rmse, 6),
                "improvement_pct": round(improvement_pct, 2),
                "n_samples": n,
            },
        )

        return result

    def compute_all_results(self) -> Dict[int, ABTestResult]:
        """Calcula resultados para todos los sensores con datos.

        Returns:
            Dict de sensor_id → ABTestResult.  Sensores con pocas
            muestras se omiten.
        """
        with self._lock:
            sensor_ids = list(self._raw_data.keys())

        results: Dict[int, ABTestResult] = {}
        for sid in sensor_ids:
            result = self.compute_results(sid)
            if result is not None:
                results[sid] = result

        return results

    def get_summary(self) -> dict:
        """Retorna resumen global del A/B testing.

        Returns:
            Dict con estadísticas agregadas.  Si no hay datos,
            retorna ``{"status": "no_data"}``.
        """
        with self._lock:
            if not self._results:
                return {"status": "no_data"}

            results = list(self._results.values())

        taylor_wins = sum(1 for r in results if r.winner == "taylor")
        baseline_wins = sum(1 for r in results if r.winner == "baseline")
        ties = sum(1 for r in results if r.winner == "tie")

        avg_taylor_mae = sum(r.taylor_mae for r in results) / len(results)
        avg_baseline_mae = sum(r.baseline_mae for r in results) / len(results)
        avg_improvement = sum(r.improvement_pct for r in results) / len(results)
        total_samples = sum(r.n_samples for r in results)

        return {
            "status": "active",
            "total_sensors": len(results),
            "taylor_wins": taylor_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
            "avg_taylor_mae": round(avg_taylor_mae, 6),
            "avg_baseline_mae": round(avg_baseline_mae, 6),
            "avg_improvement_pct": round(avg_improvement, 2),
            "total_samples": total_samples,
        }

    def get_sensor_result(self, sensor_id: int) -> Optional[ABTestResult]:
        """Retorna el último resultado calculado para un sensor.

        Args:
            sensor_id: ID del sensor.

        Returns:
            ``ABTestResult`` o ``None``.
        """
        with self._lock:
            return self._results.get(sensor_id)

    def clear(self) -> None:
        """Limpia todos los datos y resultados."""
        with self._lock:
            self._raw_data.clear()
            self._results.clear()
        logger.info("ab_test_cleared")

    def clear_sensor(self, sensor_id: int) -> None:
        """Limpia datos de un sensor específico.

        Args:
            sensor_id: ID del sensor.
        """
        with self._lock:
            self._raw_data.pop(sensor_id, None)
            self._results.pop(sensor_id, None)


# --- Singleton global ---
_global_tester: Optional[ABTester] = None
_tester_lock: threading.Lock = threading.Lock()


def get_ab_tester() -> ABTester:
    """Retorna la instancia global del A/B tester (lazy singleton).

    Returns:
        ``ABTester`` global.
    """
    global _global_tester
    if _global_tester is None:
        with _tester_lock:
            if _global_tester is None:
                _global_tester = ABTester()
    return _global_tester


def reset_ab_tester() -> None:
    """Resetea el singleton (para testing)."""
    global _global_tester
    with _tester_lock:
        if _global_tester is not None:
            _global_tester.clear()
        _global_tester = None
