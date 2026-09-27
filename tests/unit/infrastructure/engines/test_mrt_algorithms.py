"""Unit tests for MRT Algorithmic Core Modules (ZENIN v2.4+).

Verifies:
- MaxwellCurlField calculates real circulation vorticity ‖∇ × B‖.
- RamanujanCrystal guarantees contractive Tr(J*4D) < 0 and hydraulic dissipation.
- HopfSpinorField computes exact Stokes parameters without transcendentals.
- MRTPipeline orchestrates multi-step rebound trajectories.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from iot_machine_learning.infrastructure.ml.engines.mrt.algorithms import (
    HopfSpinorField,
    MaxwellCurlField,
    MRTPipeline,
    RamanujanCrystal,
)


def test_maxwell_curl_field_extraction() -> None:
    """MaxwellCurlField computes exact curl norm and phase circulation."""
    curl_field = MaxwellCurlField()
    j_test = np.array([
        [0.0, -3.0, 2.0],
        [3.0, 0.0, -1.0],
        [-2.0, 1.0, 0.0],
    ])
    omega = curl_field.extract_vorticity_tensor(j_test)
    assert np.allclose(omega, -omega.T)

    curl_norm = curl_field.compute_curl_norm(j_test)
    assert curl_norm > 0.0

    curl_vec = curl_field.compute_curl_vector_3d(j_test)
    assert curl_vec.shape == (3,)
    assert curl_vec[0] == pytest.approx(2.0)


def test_ramanujan_crystal_dissipation_invariants() -> None:
    """RamanujanCrystal guarantees negative trace Tr(J*4D) < 0 and bounded damping."""
    crystal = RamanujanCrystal(tikhonov_delta=1e-4)
    j_3d = np.array([
        [-1.0, 0.2, 0.0],
        [0.1, -1.5, 0.3],
        [0.0, 0.1, -0.8],
    ])
    j_4d = crystal.build_augmented_4d_tensor(j_3d, deformation_rate=5.0, state_3d=np.array([1.0, 0.5, 0.2]))
    assert j_4d.shape == (4, 4)

    damping, safe_trace = crystal.evaluate_crystal_dissipation(j_4d)
    assert safe_trace < 0.0
    assert 0.0 < damping <= 1.0


def test_hopf_spinor_field_exact_rational_evaluation() -> None:
    """HopfSpinorField evaluates Stokes invariants and sovereign certainty."""
    spinor_field = HopfSpinorField()
    pole = spinor_field.evaluate_pole_z2(
        deformation_rate=10.0,
        phase_theta2=0.5,
        crystal_dissipation=0.8,
        nominal_certainty=0.9,
    )
    assert 0.0 <= pole.z_amplitude <= 1.0
    assert math.isclose(pole.real_part**2 + pole.imag_part**2, pole.energy_norm_sq, abs_tol=1e-7)

    c_sov, s0, s1, s3 = spinor_field.evaluate_stokes_sovereign(
        z1_amp=0.8,
        z2_amp=pole.z_amplitude,
        phase_delta=0.0,
    )
    assert c_sov == pytest.approx(s3 + s1)
    assert s0 == pytest.approx(0.8**2 + pole.z_amplitude**2)


def test_mrt_pipeline_multi_step_rebound_trajectory() -> None:
    """MRTPipeline generates decaying multi-step rebound wave."""
    pipeline = MRTPipeline(horizon_steps=5, elasticity_rebound=1.0)
    values = [100.0, 92.0, 80.0, 65.0]

    result = pipeline.process(values, delta_time=1.0)
    assert result.status == "ok"
    assert len(result.rebound_trajectory) == 5
    # Since prices were falling, first rebound step must bounce upward
    assert result.rebound_target > 65.0
    assert result.rebound_direction == "up"
    assert result.z2_amplitude > 0.0
    assert result.ramanujan_trace < 0.0
