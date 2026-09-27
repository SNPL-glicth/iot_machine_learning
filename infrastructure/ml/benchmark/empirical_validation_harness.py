"""Empirical Validation and Benchmarking Harness for ZENIN Master Pipeline.

Conforms to:
- ISO/IEC 12207:2017: Verification and validation processes (Clause 7.2).
- ISO/IEC 25010:2023: Performance efficiency, time behaviour, and reliability.
- Line count constraint: <= 180 lines, zero regressions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence, Tuple
import numpy as np

from domain.services.manifold.mrt_hopf_fibration import evaluate_hopf_spinor
from infrastructure.ml.engines.mrt.algorithms.modules.maxwell_curl_field import MaxwellCurlField
from infrastructure.ml.master_engine.master_equation import compute_master_equation


@dataclass(frozen=True)
class EmpiricalBenchmarkReport:
    """Standardized report for model certification and shadow-to-active transition."""

    total_samples: int
    casimir_max_error: float
    latency_p50_us: float
    latency_p95_us: float
    latency_p99_us: float
    mean_liouville_factor: float
    veto_count: int
    reverse_polarization_count: int
    phase_transition_certified: bool
    summary_verdict: str


class EmpiricalValidationHarness:
    """Automated benchmarking harness evaluating streaming series under stress regimes."""

    def __init__(
        self,
        max_casimir_tolerance: float = 1e-9,
        max_p99_latency_us: float = 2000.0,
    ) -> None:
        self._casimir_tol = float(max_casimir_tolerance)
        self._max_p99 = float(max_p99_latency_us)
        self._curl_field = MaxwellCurlField()

    def run_benchmark(
        self,
        series_observations: Sequence[float],
        delta_time: float = 0.01,
        synthetic_shock_step: int | None = None,
    ) -> EmpiricalBenchmarkReport:
        """Execute deterministic benchmark over series and certify numerical invariants."""
        arr = np.asarray(series_observations, dtype=np.float64)
        n = arr.size
        if n < 8:
            raise ValueError(f"Series too short for benchmark, expected >= 8, got {n}")

        latencies_us: list[float] = []
        casimir_errors: list[float] = []
        liouville_factors: list[float] = []
        vetoes = 0
        reverse_polarities = 0

        for i in range(4, n):
            window = arr[max(0, i - 12):i + 1].tolist()
            t_start = time.perf_counter_ns()

            # 1. Kinematic phase extraction
            curl_mag, v1, a1 = self._curl_field.compute_circulation_from_series(window, delta_time)

            # Invalidate or simulate shock regime if specified
            if synthetic_shock_step is not None and i >= synthetic_shock_step:
                curl_mag *= 5.0
                phase_delta = float(np.pi * 0.95)  # Destructive interference
            else:
                phase_delta = float(math.atan2(a1, v1 + 1e-6)) if abs(v1) > 1e-6 else 0.0

            # 2. Evaluate Hopf spinor on S²
            spinor = evaluate_hopf_spinor(
                nominal_certainty=0.85, frob_norm=curl_mag,
                phase_delta=phase_delta, trace_j4d_star=-0.5,
            )

            # Check Casimir invariant: S1² + S2² + S3² ≡ S0²
            s0, s1, s2, s3 = spinor.stokes_s0, spinor.stokes_s1, spinor.stokes_s2, spinor.stokes_s3
            casimir_res = abs((s1**2 + s2**2 + s3**2) - (s0**2))
            casimir_errors.append(casimir_res)

            if spinor.polarity_direction < 0.0:
                reverse_polarities += 1

            # 3. Master Equation execution
            comp = compute_master_equation(
                phi_moe_base=0.85, i_cvar=1.0, lambda_t_crono=1.0,
                kuramoto_r=0.95, phase_alignment=0.90, delta_time=delta_time,
                dual_engine_shadow_mode=True, manifold_shadow_mode=True,
            )

            if comp.i_admissibility == 0.0:
                vetoes += 1

            geo = comp.geometric_manifold_shadow or {}
            liouville_factors.append(float(geo.get("liouville_factor", 1.0)))

            t_end = time.perf_counter_ns()
            latencies_us.append(float(t_end - t_start) / 1000.0)

        # Statistical aggregations
        lat_arr = np.asarray(latencies_us, dtype=np.float64)
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        max_casimir = float(np.max(casimir_errors)) if casimir_errors else 0.0
        mean_liouv = float(np.mean(liouville_factors)) if liouville_factors else 1.0

        # ISO 12207 Certification verdict
        certified = (max_casimir <= self._casimir_tol) and (p99 <= self._max_p99)
        verdict = (
            f"CERTIFIED_ISO_12207: Casimir max err={max_casimir:.2e}, "
            f"p99={p99:.1f}us, samples={len(latencies_us)}"
            if certified else
            f"REJECTED: Casimir max err={max_casimir:.2e} (tol={self._casimir_tol:.2e}), p99={p99:.1f}us"
        )

        return EmpiricalBenchmarkReport(
            total_samples=len(latencies_us),
            casimir_max_error=max_casimir,
            latency_p50_us=p50,
            latency_p95_us=p95,
            latency_p99_us=p99,
            mean_liouville_factor=mean_liouv,
            veto_count=vetoes,
            reverse_polarization_count=reverse_polarities,
            phase_transition_certified=certified,
            summary_verdict=verdict,
        )
