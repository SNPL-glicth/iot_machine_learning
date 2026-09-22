"""Unit tests for Kuramoto phase coherence and wave interference in Master Equation."""
from __future__ import annotations

import math
import pytest

from iot_machine_learning.infrastructure.ml.master_engine.master_equation import (
    MasterEquationComponents,
    compute_certeza,
    compute_master_equation,
)


def test_master_equation_components_kuramoto_defaults():
    comp = MasterEquationComponents(
        phi_moe_base=0.85,
        i_cvar=1.0,
        lambda_t_crono=0.90,
        phi_redrose=0.765,
    )
    assert comp.kuramoto_r == 1.0
    assert comp.phase_alignment == 1.0


def test_compute_master_equation_with_kuramoto():
    res = compute_master_equation(
        phi_moe_base=0.80,
        i_cvar=0.90,
        lambda_t_crono=0.85,
        kuramoto_r=0.90,
        phase_alignment=0.95,
    )
    # Expected: 0.80 * 0.90 * 0.85 * (0.90 * 0.95)
    expected = 0.80 * 0.90 * 0.85 * (0.90 * 0.95)
    assert res.phi_redrose == pytest.approx(expected, abs=1e-9)
    assert res.kuramoto_r == 0.90
    assert res.phase_alignment == 0.95


def test_compute_certeza_dynamic_phase_dephasing():
    # Coherent / stationary case (no velocity differential)
    c_sync = compute_certeza(
        i_cvar=1.0,
        ds_dt=0.01,
        dr_dt=0.01,
        certeza_epistemica=0.80,
    )
    assert c_sync == pytest.approx(0.80, abs=1e-9)

    # Dynamic dephasing: large velocity differential causes phase dispersion (r < 1.0)
    c_dephased = compute_certeza(
        i_cvar=1.0,
        ds_dt=0.05,
        dr_dt=0.01,
        certeza_epistemica=0.80,
    )
    # Dephased certainty should be strictly bounded in (0.0, c_sync)
    assert 0.0 < c_dephased < c_sync
