"""Unit tests for ZENIN v2.2 Master Equation pure functions and Active Mode control."""

from __future__ import annotations

import numpy as np
import pytest

from iot_machine_learning.infrastructure.ml.adapters import (
    KalmanExpertAdapter,
    RiskEngineAdapter,
    StatisticalExpertAdapter,
    TaylorExpertAdapter,
    TemporalEngineAdapter,
)
from iot_machine_learning.infrastructure.ml.engines.kalman.engine import KalmanPredictionEngine
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.engine import RosaRojaEngine
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.module1_ingestion import (
    MahalanobisFilter,
)
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.module3_moe_gating import (
    MultiplicativeMoEGating,
)
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.rhythm_generator import (
    RhythmTrajectoryGenerator,
)
from iot_machine_learning.infrastructure.ml.engines.statistical import StatisticalPredictionEngine
from iot_machine_learning.infrastructure.ml.engines.taylor.engine import TaylorPredictionEngine
from iot_machine_learning.infrastructure.ml.master_engine import (
    MasterEquationOrchestrator,
    compute_certeza,
    compute_magnitud_objetivo,
    compute_momentum_veto,
)


def _build_test_engine() -> RosaRojaEngine:
    ingestion = MahalanobisFilter(noise_threshold=3.0, history_window=50, min_samples_for_cov=10)
    rhythm = RhythmTrajectoryGenerator(min_trajectory_len=11, max_trajectory_len=15, top_k=4)
    gating = MultiplicativeMoEGating(variance_penalty=0.5)
    jury = [
        TaylorExpertAdapter(engine=TaylorPredictionEngine()),
        KalmanExpertAdapter(engine=KalmanPredictionEngine()),
        StatisticalExpertAdapter(engine=StatisticalPredictionEngine()),
    ]
    return RosaRojaEngine(
        ingestion_filter=ingestion,
        rhythm_generator=rhythm,
        moe_gating=gating,
        expert_jury=jury,
        drift_sensors=[],
    )


class TestZeninV22MasterEquation:
    """Validates ZENIN v2.2 pure mathematical formulas."""

    def test_compute_certeza_division_protection_and_veto(self) -> None:
        """Confirms division-by-zero protection and hard CVaR veto behavior."""
        # 1. Zero division protection: dr_dt = 0.0 with sigma_dr
        c_protected = compute_certeza(
            i_cvar=1.0,
            ds_dt=0.005,
            dr_dt=0.0,
            certeza_epistemica=0.85,
            sigma_dr=1.0,
            epsilon=1e-6,
        )
        assert 0.0 <= c_protected <= 1.0

        # 2. Hard risk veto: i_cvar = 0.0 forces certainty to 0.0
        c_veto = compute_certeza(
            i_cvar=0.0,
            ds_dt=0.01,
            dr_dt=0.01,
            certeza_epistemica=0.90,
        )
        assert c_veto == 0.0

        # 3. Exact synchrony: ds_dt == dr_dt implies ratio == 1.0, Lambda == 1.0
        c_sync = compute_certeza(
            i_cvar=1.0,
            ds_dt=0.02,
            dr_dt=0.02,
            certeza_epistemica=0.80,
        )
        assert c_sync == pytest.approx(0.80, abs=1e-9)

    def test_compute_magnitud_objetivo_weighted_average(self) -> None:
        """Verifies pure weighted average in market units."""
        # Predictions: Taylor = 0.0040, Kalman = 0.0030, Statistical = 0.0020
        preds = [0.0040, 0.0030, 0.0020]
        weights = [1.2, 1.0, 0.8]  # Sum = 3.0
        # (1.2 * 0.0040 + 1.0 * 0.0030 + 0.8 * 0.0020) / 3.0 = 0.0094 / 3.0 = 0.0031333
        expected = (1.2 * 0.0040 + 1.0 * 0.0030 + 0.8 * 0.0020) / 3.0
        mag = compute_magnitud_objetivo(preds, weights)
        assert mag == pytest.approx(expected, abs=1e-7)

        # Empty fallback
        assert compute_magnitud_objetivo([], default=0.001) == 0.001

    def test_compute_momentum_veto_deadband(self) -> None:
        """Confirms Heaviside momentum deadband filter."""
        # Aligned momentum exceeding deadband:
        # ds_dt_ema * mag = 0.01 * 0.005 = 0.00005
        # tau_mom * sigma_mom = 0.5 * 0.00005 = 0.000025 -> signal > 0
        v_pass = compute_momentum_veto(ds_dt_ema=0.01, magnitud=0.005, tau_mom=0.5, sigma_mom=0.00005)
        assert v_pass == 1.0

        # Counter-momentum: negative product -> veto (0.0)
        v_counter = compute_momentum_veto(ds_dt_ema=-0.01, magnitud=0.005, tau_mom=0.5, sigma_mom=0.00005)
        assert v_counter == 0.0

        # Weak momentum within noise deadband: signal < 0 -> veto (0.0)
        v_noise = compute_momentum_veto(ds_dt_ema=0.0001, magnitud=0.0001, tau_mom=2.0, sigma_mom=0.001)
        assert v_noise == 0.0

    def test_active_mode_full_control_and_real_magnitude(self) -> None:
        """Verifies that when shadow_mode=False, Master Equation assumes full control."""
        engine = _build_test_engine()
        orch = MasterEquationOrchestrator(
            rosa_roja_engine=engine,
            risk_adapter=RiskEngineAdapter(default_sigma=0.001),
            temporal_adapter=TemporalEngineAdapter(),
            shadow_mode=False,  # Active control enabled!
            tau_mom=0.1,
            sigma_mom=0.0001,
        )

        # Warm up Mahalanobis and trajectory tracker
        rng = np.random.default_rng(123)
        for _ in range(15):
            d = rng.normal(0, 0.001, size=10)
            d[0] = 0.004
            orch.process_event(d, 1.0)

        # Active evaluation step
        delta = rng.normal(0, 0.001, size=10)
        delta[0] = 0.005
        plan = orch.process_event(delta, 1.0)

        trace = plan.envelope.metadata["decision_trace"] if plan.envelope else plan.veto_details["decision_trace"]
        assert trace["execution_mode"] == "active"
        assert trace["governing_component"] == "master_equation"

        if plan.action == "EXECUTE":
            assert plan.envelope is not None
            # Real target magnitude in envelope, not a static score
            assert plan.envelope.magnitude > 0.0
            assert plan.global_confidence == pytest.approx(trace["certeza"], abs=1e-9)
