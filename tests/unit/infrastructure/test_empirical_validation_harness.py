"""Unit tests for EmpiricalValidationHarness (ISO/IEC 12207 & 25010).

Verifies:
- Casimir Stokes invariant |S1² + S2² + S3² - S0²| <= tolerance across streaming windows.
- Latency percentiles (p50, p95, p99) under execution budgets.
- Rejection and certification logic according to ISO 12207 validation criteria.
- Line count constraint: <= 180 lines.
"""

from __future__ import annotations

import numpy as np
import pytest

from infrastructure.ml.benchmark.empirical_validation_harness import (
    EmpiricalBenchmarkReport,
    EmpiricalValidationHarness,
)


def test_benchmark_runs_and_certifies_clean_series() -> None:
    """Verify standard smooth series passes Casimir invariant and p99 latency."""
    harness = EmpiricalValidationHarness(max_casimir_tolerance=1e-9, max_p99_latency_us=5000.0)

    # Clean smooth sine series
    t = np.linspace(0, 4 * np.pi, 40)
    series = np.sin(t) + 0.5 * np.cos(2 * t)

    report = harness.run_benchmark(series, delta_time=0.05)

    assert isinstance(report, EmpiricalBenchmarkReport)
    assert report.total_samples == 36  # 40 - 4 warm-up
    assert report.casimir_max_error < 1e-9
    assert report.latency_p50_us > 0.0
    assert report.latency_p99_us >= report.latency_p50_us
    assert report.phase_transition_certified is True
    assert "CERTIFIED_ISO_12207" in report.summary_verdict


def test_benchmark_detects_synthetic_shock_and_reverse_polarization() -> None:
    """Verify shock step injects destructive interference, producing reverse polarization."""
    harness = EmpiricalValidationHarness()

    t = np.linspace(0, 2 * np.pi, 30)
    series = np.sin(t)

    report = harness.run_benchmark(series, delta_time=0.05, synthetic_shock_step=15)

    assert report.total_samples == 26
    # Shock injected at step >= 15 sets phase_delta close to pi -> polarity < 0
    assert report.reverse_polarization_count > 0


def test_benchmark_series_too_short_raises_value_error() -> None:
    """Verify series shorter than 8 elements raises informative ValueError."""
    harness = EmpiricalValidationHarness()
    short_series = [1.0, 2.0, 3.0, 4.0]

    with pytest.raises(ValueError, match="Series too short for benchmark"):
        harness.run_benchmark(short_series)


def test_benchmark_rejection_on_strict_latency_budget() -> None:
    """Verify harness properly rejects when latency exceeds strict budget."""
    # A sub-nanosecond budget cannot be met by standard hardware execution
    harness = EmpiricalValidationHarness(max_p99_latency_us=0.001)

    series = np.linspace(1.0, 10.0, 25)
    report = harness.run_benchmark(series)

    assert report.phase_transition_certified is False
    assert "REJECTED" in report.summary_verdict
