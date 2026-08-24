#!/usr/bin/env python
"""Benchmark: Rosa Roja Engine process_event() latency.

Compares IoT (chiller/CA) vs Market (microstructure) scenarios.
Measures P50, P95, P99 latency in sub-milliseconds.
"""

from __future__ import annotations

import time
import statistics
import random
import logging
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

# Suppress Mahalanobis covariance warnings during benchmark
logging.getLogger("core.orchestration.rosa_roja.modules.module1_ingestion").setLevel(logging.ERROR)

from core.orchestration.rosa_roja.domain.movement import Movement, RhythmSignature
from core.orchestration.rosa_roja.domain.trajectory import Trajectory, TerminalState
from core.orchestration.rosa_roja.domain.validation import ValidationResult, VetoDetails
from core.orchestration.rosa_roja.domain.execution import ExecutionPlan, ActionEnvelope
from core.orchestration.rosa_roja.modules.module1_ingestion import MahalanobisFilter
from core.orchestration.rosa_roja.modules.rhythm_generator import RhythmTrajectoryGenerator
from core.orchestration.rosa_roja.modules.module3_moe_gating import MultiplicativeMoEGating
from core.orchestration.rosa_roja.ports.expert_jury import ExpertJuryPort
from core.orchestration.rosa_roja.ports.drift_sensor import DriftSensorPort
from core.orchestration.rosa_roja.engine import RosaRojaEngine


# ============================================================================
# Mock Components
# ============================================================================

class BenchmarkExpertJuryPort:
    """High-performance mock expert for benchmarking."""
    
    def __init__(self, name: str, is_critical: bool = False, threshold: float = 0.5, weight: float = 1.0, score: float = 0.8):
        self.name = name
        self.is_critical = is_critical
        self.threshold = threshold
        self.weight = weight
        self._score = score
    
    def evaluate_trajectory(self, trajectory: Trajectory) -> float:
        return self._score
    
    def update_learning(self, actual: float, predicted: float) -> None:
        pass


class BenchmarkDriftSensorPort:
    """High-performance mock drift sensor for benchmarking."""
    
    def __init__(self, name: str, drift_score: float = 0.1):
        self.name = name
        self._drift_score = drift_score
    
    def get_drift_score(self) -> float:
        return self._drift_score
    
    def update(self, actual: float, predicted: float) -> None:
        pass
    
    def reset(self) -> None:
        pass


# ============================================================================
# Scenario Generators
# ============================================================================

def generate_market_delta_state() -> np.ndarray:
    """Generate typical market microstructure delta_state (3-5 dims).
    
    Typical: [price_change, volume_delta, bid_ask_spread, order_flow_imbalance, volatility]
    """
    return np.array([
        np.random.normal(0, 0.001),      # price_change (bps)
        np.random.normal(0, 100),        # volume_delta
        np.random.exponential(0.01),     # bid_ask_spread
        np.random.normal(0, 0.1),        # order_flow_imbalance
        np.random.exponential(0.001),    # volatility
    ])


def generate_iot_delta_state() -> np.ndarray:
    """Generate typical IoT (chiller/CA) delta_state (8-12 dims).
    
    Typical: [temp_in, temp_out, pressure_suction, pressure_discharge, 
              compressor_speed, power_consumption, vibration, oil_temp, ...]
    """
    return np.array([
        np.random.normal(12, 2),         # chilled_water_in_temp (°C)
        np.random.normal(7, 1.5),        # chilled_water_out_temp (°C)
        np.random.normal(4, 0.5),        # suction_pressure (bar)
        np.random.normal(12, 1),         # discharge_pressure (bar)
        np.random.normal(3000, 100),     # compressor_speed (RPM)
        np.random.normal(150, 20),       # power_consumption (kW)
        np.random.exponential(0.5),      # vibration (mm/s)
        np.random.normal(50, 5),         # oil_temp (°C)
        np.random.normal(35, 3),         # condenser_temp (°C)
        np.random.normal(40, 2),         # ambient_temp (°C)
        np.random.normal(0.8, 0.1),      # cop (coefficient of performance)
        np.random.normal(0.95, 0.02),    # efficiency_ratio
    ])


@dataclass
class BenchmarkResult:
    scenario: str
    dims: int
    iterations: int
    warmup: int
    p50_us: float
    p95_us: float
    p99_us: float
    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    throughput_per_sec: float


def run_benchmark(
    scenario_name: str,
    delta_state_generator,
    dims: int,
    iterations: int = 500,
    warmup: int = 100,
) -> BenchmarkResult:
    """Run benchmark for a specific scenario."""
    
    # Setup engine with minimal configuration for latency measurement
    ingestion = MahalanobisFilter(noise_threshold=3.0, history_window=50, min_samples_for_cov=10)
    rhythm = RhythmTrajectoryGenerator(
        min_trajectory_len=11,
        max_trajectory_len=15,
        top_k=3,
        oversample_factor=2,
        quantization_decimals=2,
    )
    gating = MultiplicativeMoEGating(variance_penalty=0.5)
    
    # 3 experts (2 critical, 1 non-critical) - typical production setup
    experts = [
        BenchmarkExpertJuryPort("taylor", True, 0.65, 1.2, 0.8),
        BenchmarkExpertJuryPort("kalman", True, 0.60, 1.0, 0.85),
        BenchmarkExpertJuryPort("statistical", False, 0.55, 0.8, 0.75),
    ]
    
    sensors = [
        BenchmarkDriftSensorPort("page_hinkley", 0.1),
        BenchmarkDriftSensorPort("adwin", 0.05),
    ]
    
    engine = RosaRojaEngine(
        ingestion_filter=ingestion,
        rhythm_generator=rhythm,
        moe_gating=gating,
        expert_jury=experts,
        drift_sensors=sensors,
    )
    
    # Warmup - build history for trajectory generation
    print(f"  Warming up {scenario_name} ({warmup} iterations)...")
    for _ in range(warmup):
        delta_state = delta_state_generator()
        delta_time = 1.0
        engine.process_event(delta_state, delta_time)
    
    # Benchmark
    print(f"  Benchmarking {scenario_name} ({iterations} iterations)...")
    latencies = []
    
    for i in range(iterations):
        delta_state = delta_state_generator()
        delta_time = 1.0
        
        start = time.perf_counter_ns()
        engine.process_event(delta_state, delta_time)
        end = time.perf_counter_ns()
        
        latencies.append((end - start) / 1000.0)  # Convert to microseconds
        
        if (i + 1) % 100 == 0:
            print(f"    Completed {i + 1}/{iterations}")
    
    latencies = np.array(latencies)
    
    return BenchmarkResult(
        scenario=scenario_name,
        dims=dims,
        iterations=iterations,
        warmup=warmup,
        p50_us=float(np.percentile(latencies, 50)),
        p95_us=float(np.percentile(latencies, 95)),
        p99_us=float(np.percentile(latencies, 99)),
        mean_us=float(np.mean(latencies)),
        std_us=float(np.std(latencies)),
        min_us=float(np.min(latencies)),
        max_us=float(np.max(latencies)),
        throughput_per_sec=iterations / (np.sum(latencies) / 1_000_000),
    )


def print_results(results: List[BenchmarkResult]) -> None:
    """Print formatted benchmark results."""
    print("\n" + "=" * 100)
    print("ROSA ROJA ENGINE - process_event() LATENCY BENCHMARK")
    print("=" * 100)
    print(f"{'Scenario':<20} {'Dims':<6} {'Iter':<8} {'P50 (μs)':<10} {'P95 (μs)':<10} {'P99 (μs)':<10} {'Mean (μs)':<10} {'Std (μs)':<10} {'Throughput/s':<12}")
    print("-" * 100)
    
    for r in results:
        print(f"{r.scenario:<20} {r.dims:<6} {r.iterations:<8} {r.p50_us:<10.1f} {r.p95_us:<10.1f} {r.p99_us:<10.1f} {r.mean_us:<10.1f} {r.std_us:<10.1f} {r.throughput_per_sec:<12.0f}")
    
    print("-" * 100)
    print(f"\nSub-millisecond target: {'✅ ACHIEVED' if all(r.p99_us < 1000 for r in results) else '❌ MISSED'}")
    print(f"All P99 < 1ms: {all(r.p99_us < 1000 for r in results)}")


def main():
    """Run benchmarks for both scenarios."""
    print("Rosa Roja Engine Benchmark")
    print("=" * 50)
    
    results = []
    
    # Market microstructure scenario (5 dims)
    results.append(run_benchmark(
        scenario_name="Market (5D)",
        delta_state_generator=generate_market_delta_state,
        dims=5,
        iterations=10000,
        warmup=2000,
    ))
    
    # IoT chiller/CA scenario (12 dims)
    results.append(run_benchmark(
        scenario_name="IoT Chiller (12D)",
        delta_state_generator=generate_iot_delta_state,
        dims=12,
        iterations=10000,
        warmup=2000,
    ))
    
    # Higher-dimensional IoT (20 dims) - stress test
    def generate_iot_20d():
        base = generate_iot_delta_state()
        extra = np.random.normal(0, 1, 8)
        return np.concatenate([base, extra])
    
    results.append(run_benchmark(
        scenario_name="IoT Extended (20D)",
        delta_state_generator=generate_iot_20d,
        dims=20,
        iterations=5000,
        warmup=1000,
    ))
    
    print_results(results)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for r in results:
        print(f"\n{r.scenario} ({r.dims}D):")
        print(f"  P50: {r.p50_us:.1f} μs  |  P95: {r.p95_us:.1f} μs  |  P99: {r.p99_us:.1f} μs")
        print(f"  Mean: {r.mean_us:.1f} μs ± {r.std_us:.1f} μs")
        print(f"  Throughput: {r.throughput_per_sec:,.0f} events/sec")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main()