"""Medición rigurosa de recursos computacionales para transformaciones de señal.

Registra:
- Tiempo de ejecución (ms)
- Memoria máxima asignada (KB vía tracemalloc)
- Throughput (puntos/segundo)
- Latencia unitaria por punto (μs)
- Factor de compresión (N_in / N_out)
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .transformations import TransformedSignal


@dataclass(frozen=True)
class ResourceBenchmarkResult:
    """Métricas de costo computacional de una representación."""

    name: str
    n_input_points: int
    n_output_points: int
    compression_ratio: float
    elapsed_ms: float
    peak_memory_kb: float
    throughput_pts_per_sec: float
    latency_per_point_us: float


def benchmark_transformation(
    transform_fn: Callable[..., TransformedSignal],
    values: np.ndarray,
    timestamps_sec: np.ndarray,
    timestamps_raw: list[str],
    n_warmup: int = 1,
    n_runs: int = 3,
    **kwargs: Any,
) -> tuple[TransformedSignal, ResourceBenchmarkResult]:
    """Ejecuta y perfila una función de transformación con warmup y promediado."""
    # Warmup
    for _ in range(n_warmup):
        _ = transform_fn(values, timestamps_sec, timestamps_raw, **kwargs)

    # Medición de memoria y tiempo
    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0]

    times_ns: list[int] = []
    signal: TransformedSignal | None = None

    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        signal = transform_fn(values, timestamps_sec, timestamps_raw, **kwargs)
        t1 = time.perf_counter_ns()
        times_ns.append(t1 - t0)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert signal is not None

    avg_elapsed_ms = (sum(times_ns) / len(times_ns)) / 1_000_000.0
    mem_allocated_kb = max(0.0, (peak_mem - start_mem) / 1024.0)
    n_in = len(values)
    n_out = signal.n_points
    throughput = (n_in / (avg_elapsed_ms / 1000.0)) if avg_elapsed_ms > 0 else 0.0
    latency_per_pt_us = (avg_elapsed_ms * 1000.0 / n_in) if n_in > 0 else 0.0

    result = ResourceBenchmarkResult(
        name=signal.name,
        n_input_points=n_in,
        n_output_points=n_out,
        compression_ratio=signal.compression_ratio,
        elapsed_ms=round(avg_elapsed_ms, 3),
        peak_memory_kb=round(mem_allocated_kb, 2),
        throughput_pts_per_sec=round(throughput, 1),
        latency_per_point_us=round(latency_per_pt_us, 4),
    )

    return signal, result
