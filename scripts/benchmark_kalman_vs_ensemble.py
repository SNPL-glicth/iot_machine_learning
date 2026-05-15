"""Benchmark: KalmanPredictionEngine vs Statistical, Baseline, Taylor.

Genera series sintéticas con ruido controlado y compara RMSE, MAE,
y cobertura de intervalos de confianza.

Usage:
    PYTHONPATH=/home/nicolas/Documentos/Iot_System/iot_machine_learning:/home/nicolas/Documentos/Iot_System \
    /home/nicolas/Documentos/Iot_System/.venv/bin/python \
    scripts/benchmark_kalman_vs_ensemble.py
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from iot_machine_learning.infrastructure.ml.engines import (
    BaselineMovingAverageEngine,
    EngineFactory,
    KalmanPredictionEngine,
    StatisticalPredictionEngine,
    TaylorPredictionEngine,
)


@dataclass
class BenchmarkResult:
    engine_name: str
    rmse: float
    mae: float
    mape: float
    coverage_95: float  # % de valores reales dentro del intervalo
    avg_confidence: float
    avg_latency_ms: float


def generate_series(
    n: int = 200,
    noise_sigma: float = 2.0,
    trend_slope: float = 0.0,
    seasonal_amplitude: float = 0.0,
    seed: int = 42,
) -> List[float]:
    """Genera serie sintética: base + tendencia + estacionalidad + ruido."""
    random.seed(seed)
    np.random.seed(seed)
    base = 50.0
    values = []
    for i in range(n):
        trend = trend_slope * i
        seasonal = seasonal_amplitude * math.sin(2 * math.pi * i / 24)
        noise = random.gauss(0, noise_sigma)
        values.append(base + trend + seasonal + noise)
    return values


def run_walk_forward(
    engine,
    values: List[float],
    window_size: int = 30,
    horizon: int = 1,
) -> Tuple[List[float], List[float], List[float], List[Tuple[float, float]]]:
    """Walk-forward validation: predice el siguiente punto y compara."""
    predictions = []
    actuals = []
    confidences = []
    intervals = []

    for i in range(window_size, len(values) - horizon + 1):
        window = values[i - window_size : i]
        actual = values[i + horizon - 1]

        t0 = time.perf_counter()
        result = engine.predict(window)
        latency_ms = (time.perf_counter() - t0) * 1000

        predictions.append(result.predicted_value)
        actuals.append(actual)
        confidences.append(result.confidence)

        # Extraer intervalo si existe
        interval = result.metadata.get("confidence_interval")
        if interval and isinstance(interval, tuple) and len(interval) == 2:
            intervals.append(interval)
        else:
            # Fallback: intervalo simétrico basado en confidence heuristic
            std_est = abs(result.predicted_value - actual) / (result.confidence + 0.01)
            intervals.append((
                result.predicted_value - 1.96 * std_est,
                result.predicted_value + 1.96 * std_est,
            ))

    return predictions, actuals, confidences, intervals


def evaluate(
    engine_name: str,
    predictions: List[float],
    actuals: List[float],
    confidences: List[float],
    intervals: List[Tuple[float, float]],
    latencies: List[float],
) -> BenchmarkResult:
    """Calcula métricas de error."""
    errors = [p - a for p, a in zip(predictions, actuals)]
    abs_errors = [abs(e) for e in errors]
    squared_errors = [e ** 2 for e in errors]

    rmse = math.sqrt(statistics.mean(squared_errors))
    mae = statistics.mean(abs_errors)

    # MAPE con protección contra división por cero
    mape_values = []
    for a, e in zip(actuals, abs_errors):
        if abs(a) > 1e-6:
            mape_values.append(e / abs(a) * 100)
    mape = statistics.mean(mape_values) if mape_values else float("inf")

    # Cobertura del intervalo al 95%
    covered = sum(
        1
        for a, (lo, hi) in zip(actuals, intervals)
        if lo <= a <= hi
    )
    coverage_95 = covered / len(actuals) * 100

    return BenchmarkResult(
        engine_name=engine_name,
        rmse=rmse,
        mae=mae,
        mape=mape,
        coverage_95=coverage_95,
        avg_confidence=statistics.mean(confidences),
        avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
    )


def benchmark_scenario(
    name: str,
    values: List[float],
    window_size: int = 30,
) -> Dict[str, BenchmarkResult]:
    """Ejecuta benchmark para un escenario."""
    print(f"\n{'=' * 60}")
    print(f"Escenario: {name} (n={len(values)})")
    print(f"{'=' * 60}")

    engines = {
        "kalman": KalmanPredictionEngine(warmup_size=5, horizon=1),
        "statistical": StatisticalPredictionEngine(),
        "baseline": BaselineMovingAverageEngine(),
        "taylor": TaylorPredictionEngine(),
    }

    results = {}
    for engine_name, engine in engines.items():
        preds, actuals, confs, intervals = run_walk_forward(
            engine, values, window_size=window_size
        )

        # Recolectar latencias (ya medidas dentro de run_walk_forward)
        # Re-ejecutamos una vez más para medir latencia limpia
        latencies = []
        for i in range(window_size, min(window_size + 50, len(values))):
            window = values[i - window_size : i]
            t0 = time.perf_counter()
            engine.predict(window)
            latencies.append((time.perf_counter() - t0) * 1000)

        result = evaluate(engine_name, preds, actuals, confs, intervals, latencies)
        results[engine_name] = result

        print(
            f"  {engine_name:12s} | RMSE={result.rmse:6.3f} | MAE={result.mae:6.3f} | "
            f"MAPE={result.mape:5.2f}% | Cov95={result.coverage_95:5.1f}% | "
            f"Conf={result.avg_confidence:.3f} | Lat={result.avg_latency_ms:.3f}ms"
        )

    return results


def print_improvement_table(
    results: Dict[str, BenchmarkResult],
    baseline_name: str = "baseline",
) -> None:
    """Muestra porcentaje de mejora de Kalman vs otros motores."""
    kalman = results.get("kalman")
    if not kalman:
        return

    print(f"\n{'=' * 60}")
    print(f"Mejora de Kalman vs otros motores")
    print(f"{'=' * 60}")

    for name, result in sorted(results.items()):
        if name == "kalman":
            continue
        rmse_improvement = (result.rmse - kalman.rmse) / result.rmse * 100
        mae_improvement = (result.mae - kalman.mae) / result.mae * 100
        mape_improvement = (result.mape - kalman.mape) / result.mape * 100

        print(f"  vs {name:12s} | RMSE: {rmse_improvement:+6.1f}% | "
              f"MAE: {mae_improvement:+6.1f}% | MAPE: {mape_improvement:+6.1f}%")


def main() -> None:
    print("=" * 60)
    print("BENCHMARK: KalmanPredictionEngine vs Ensemble")
    print("=" * 60)

    # Escenario 1: Ruido moderado, sin tendencia
    values1 = generate_series(n=200, noise_sigma=2.0, trend_slope=0.0, seed=42)
    results1 = benchmark_scenario("Ruido moderado (σ=2, sin tendencia)", values1)
    print_improvement_table(results1)

    # Escenario 2: Ruido alto, sin tendencia
    values2 = generate_series(n=200, noise_sigma=5.0, trend_slope=0.0, seed=43)
    results2 = benchmark_scenario("Ruido alto (σ=5, sin tendencia)", values2)
    print_improvement_table(results2)

    # Escenario 3: Ruido moderado + tendencia
    values3 = generate_series(n=200, noise_sigma=2.0, trend_slope=0.1, seed=44)
    results3 = benchmark_scenario("Ruido + tendencia (σ=2, slope=0.1)", values3)
    print_improvement_table(results3)

    # Escenario 4: Ruido alto + tendencia (caso adversarial)
    values4 = generate_series(n=200, noise_sigma=5.0, trend_slope=0.1, seed=45)
    results4 = benchmark_scenario("Ruido alto + tendencia (σ=5, slope=0.1)", values4)
    print_improvement_table(results4)

    # Escenario 5: Ruido moderado + estacionalidad
    values5 = generate_series(n=200, noise_sigma=2.0, seasonal_amplitude=5.0, seed=46)
    results5 = benchmark_scenario("Ruido + estacionalidad (σ=2, amp=5)", values5)
    print_improvement_table(results5)

    # Resumen global
    print(f"\n{'=' * 60}")
    print("RESUMEN GLOBAL: Mejora promedio de Kalman")
    print(f"{'=' * 60}")

    all_results = [results1, results2, results3, results4, results5]
    for competitor in ["baseline", "statistical", "taylor"]:
        rmse_imps = []
        mae_imps = []
        mape_imps = []
        for r in all_results:
            if "kalman" not in r or competitor not in r:
                continue
            k = r["kalman"]
            c = r[competitor]
            rmse_imps.append((c.rmse - k.rmse) / c.rmse * 100)
            mae_imps.append((c.mae - k.mae) / c.mae * 100)
            mape_imps.append((c.mape - k.mape) / c.mape * 100)

        print(
            f"  vs {competitor:12s} | RMSE: {statistics.mean(rmse_imps):+6.1f}% | "
            f"MAE: {statistics.mean(mae_imps):+6.1f}% | MAPE: {statistics.mean(mape_imps):+6.1f}%"
        )

    print(f"\n{'=' * 60}")
    print("Nota: valores positivos = Kalman es MEJOR (menor error)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
