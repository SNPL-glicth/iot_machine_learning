#!/usr/bin/env python
"""Quick benchmark: Rosa Roja Engine individual component latencies."""

from __future__ import annotations

import time
import numpy as np
import logging

logging.getLogger("core.orchestration.rosa_roja.modules.module1_ingestion").setLevel(logging.ERROR)

from core.orchestration.rosa_roja.modules.module1_ingestion import MahalanobisFilter
from core.orchestration.rosa_roja.modules.rhythm_generator import RhythmTrajectoryGenerator
from core.orchestration.rosa_roja.modules.module3_moe_gating import MultiplicativeMoEGating
from core.orchestration.rosa_roja.domain.movement import Movement, RhythmSignature
from core.orchestration.rosa_roja.domain.trajectory import Trajectory, TerminalState
from core.orchestration.rosa_roja.ports.expert_jury import ExpertJuryPort
from core.orchestration.rosa_roja.ports.drift_sensor import DriftSensorPort
from core.orchestration.rosa_roja.engine import RosaRojaEngine


class SimpleExpert:
    def __init__(self, name, is_critical, threshold, weight, score):
        self.name = name
        self.is_critical = is_critical
        self.threshold = threshold
        self.weight = weight
        self._score = score
    def evaluate_trajectory(self, traj): return self._score
    def update_learning(self, a, p): pass

class SimpleDrift:
    def __init__(self, name, score): self.name = name; self._score = score
    def get_drift_score(self): return self._score
    def update(self, a, p): pass
    def reset(self): pass


def time_it(fn, iterations=100):
    """Time a function over iterations, return stats in microseconds."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)
    arr = np.array(times)
    return {
        "mean": np.mean(arr), "p50": np.percentile(arr, 50),
        "p95": np.percentile(arr, 95), "p99": np.percentile(arr, 99),
        "std": np.std(arr), "min": np.min(arr), "max": np.max(arr)
    }


def main():
    print("=" * 70)
    print("ROSA ROJA COMPONENT BENCHMARK (quick)")
    print("=" * 70)
    
    # ---- Module 1: Mahalanobis Filter ----
    print("\n[Module 1] MahalanobisFilter.process_raw_step()")
    filter = MahalanobisFilter(noise_threshold=3.0, history_window=50, min_samples_for_cov=10)
    
    # Warmup
    for i in range(20):
        filter.process_raw_step(np.random.randn(5), 1.0)
    
    def m1_step():
        filter.process_raw_step(np.random.randn(5), 1.0)
    
    stats = time_it(m1_step, 500)
    print(f"  5D:  P50={stats['p50']:.1f}μs  P95={stats['p95']:.1f}μs  P99={stats['p99']:.1f}μs  Mean={stats['mean']:.1f}μs")
    
    filter12 = MahalanobisFilter(noise_threshold=3.0, history_window=50, min_samples_for_cov=10)
    for i in range(20):
        filter12.process_raw_step(np.random.randn(12), 1.0)
    def m1_step_12():
        filter12.process_raw_step(np.random.randn(12), 1.0)
    stats = time_it(m1_step_12, 500)
    print(f"  12D: P50={stats['p50']:.1f}μs  P95={stats['p95']:.1f}μs  P99={stats['p99']:.1f}μs  Mean={stats['mean']:.1f}μs")
    
    # ---- Module 2: Rhythm Generator (trajectory generation) ----
    print("\n[Module 2] RhythmTrajectoryGenerator.generate_candidate_trajectories()")
    rhythm = RhythmTrajectoryGenerator(min_trajectory_len=3, max_trajectory_len=5, top_k=2, oversample_factor=2)
    
    # Pre-fill history
    for i in range(10):
        m = Movement(
            delta_state=np.array([float(i), 0.0]),
            delta_time=1.0,
            velocity=1.0,
            direction=np.array([1.0, 0.0]),
            rhythm_signature=RhythmSignature(1.0, 0.0, 0.0, 0.0, 0.0),
            mahalanobis_distance=0.0,
            timestamp=float(i),
        )
        rhythm._history.append(m)
    rhythm._update_transition_graph()
    
    latest = Movement(
        delta_state=np.array([5.0, 0.0]),
        delta_time=1.0,
        velocity=1.0,
        direction=np.array([1.0, 0.0]),
        rhythm_signature=RhythmSignature(1.0, 0.0, 0.0, 0.0, 0.0),
        mahalanobis_distance=0.0,
        timestamp=5.0,
    )
    
    def m2_gen():
        rhythm.generate_candidate_trajectories(latest, 0.1)
    
    stats = time_it(m2_gen, 100)
    print(f"  P50={stats['p50']:.1f}μs  P95={stats['p95']:.1f}μs  P99={stats['p99']:.1f}μs  Mean={stats['mean']:.1f}μs")
    
    # ---- Module 3: MoE Gating ----
    print("\n[Module 3] MultiplicativeMoEGating.evaluate_and_veto()")
    gating = MultiplicativeMoEGating()
    experts = [
        SimpleExpert("taylor", True, 0.65, 1.2, 0.8),
        SimpleExpert("kalman", True, 0.60, 1.0, 0.85),
        SimpleExpert("statistical", False, 0.55, 0.8, 0.75),
    ]
    
    # Create test trajectories
    trajs = []
    for _ in range(3):
        movements = []
        for i in range(3):
            delta_state = np.array([float(i), 0.0])
            rhythm_sig = RhythmSignature(1.0, 0.0, 0.0, 0.0, 0.0)
            m = Movement(delta_state, 1.0, 1.0, np.array([1.0, 0.0]), rhythm_sig, 0.0, float(i))
            movements.append(m)
        trajs.append(Trajectory(
            movements=tuple(movements), coherence_score=0.8, invalidation_step=None,
            terminal_state=TerminalState(movements[-1].delta_state, 2, 0.8)
        ))
    
    def m3_gate():
        gating.evaluate_and_veto(trajs, experts)
    
    stats = time_it(m3_gate, 500)
    print(f"  P50={stats['p50']:.1f}μs  P95={stats['p95']:.1f}μs  P99={stats['p99']:.1f}μs  Mean={stats['mean']:.1f}μs")
    
    # ---- Full Engine (with minimal history for HOLD path) ----
    print("\n[Full Engine] RosaRojaEngine.process_event() - HOLD path (insufficient history)")
    ingestion = MahalanobisFilter(noise_threshold=3.0, history_window=50, min_samples_for_cov=10)
    rhythm = RhythmTrajectoryGenerator(min_trajectory_len=11, max_trajectory_len=15, top_k=2, oversample_factor=2)
    gating = MultiplicativeMoEGating(variance_penalty=0.5)
    experts = [SimpleExpert("taylor", True, 0.65, 1.2, 0.8), SimpleExpert("kalman", True, 0.60, 1.0, 0.85)]
    sensors = [SimpleDrift("adwin", 0.1)]
    
    engine = RosaRojaEngine(ingestion, rhythm, gating, experts, sensors)
    
    def engine_hold():
        engine.process_event(np.random.randn(5), 1.0)
    
    stats = time_it(engine_hold, 200)
    print(f"  5D (HOLD):  P50={stats['p50']:.1f}μs  P95={stats['p95']:.1f}μs  P99={stats['p99']:.1f}μs  Mean={stats['mean']:.1f}μs")
    
    # ---- Full Engine (with enough history for EXECUTE path) ----
    print("\n[Full Engine] RosaRojaEngine.process_event() - EXECUTE path (with history)")
    # Build history with 5D to match HOLD path engine
    engine2 = RosaRojaEngine(ingestion, rhythm, gating, experts, sensors)
    for i in range(20):
        engine2.process_event(np.array([float(i % 5), 0.0, 0.0, 0.0, 0.0]), 1.0)
    
    def engine_exec():
        engine2.process_event(np.array([0.0, 0.0, 0.0, 0.0, 0.0]), 1.0)
    
    stats = time_it(engine_exec, 100)
    print(f"  5D (EXECUTE): P50={stats['p50']:.1f}μs  P95={stats['p95']:.1f}μs  P99={stats['p99']:.1f}μs  Mean={stats['mean']:.1f}μs")
    
    print("\n" + "=" * 70)
    print("NOTE: Full EXECUTE path includes trajectory generation (Module 2)")
    print("which dominates latency. Module 1 + Module 3 are sub-ms.")
    print("=" * 70)


if __name__ == "__main__":
    np.random.seed(42)
    main()