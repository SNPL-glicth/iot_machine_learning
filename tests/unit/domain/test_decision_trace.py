"""Unit tests for ISO/IEC 22989 & 25010 DecisionTraceRecord."""

from __future__ import annotations

import numpy as np
import pytest

from domain.entities.rosa_roja.decision_trace import (
    DecisionTraceRecord,
    ExecutionMode,
    SubsystemStatus,
)


def test_decision_trace_record_creation_and_immutability() -> None:
    """Verifies that DecisionTraceRecord enforces immutability and valid ranges."""
    state = np.array([0.01, -0.02, 0.05], dtype=np.float64)
    record = DecisionTraceRecord.create(
        delta_state=state,
        delta_time=1.0,
        phi_moe_base=0.85,
        i_cvar=1.0,
        lambda_t_crono=0.92,
        certeza=0.78,
        magnitud_objetivo=0.015,
        momentum_veto=1.0,
        shadow_mode=False,
        action_verdict="EXECUTE",
        state_dim=3,
        mahalanobis_d=1.45,
        dynamic_threshold=3.0,
        is_outlier=False,
        subsystem_health={
            "risk_engine": SubsystemStatus.HEALTHY,
            "temporal_engine": SubsystemStatus.HEALTHY,
        },
    )

    assert record.execution_mode == ExecutionMode.ACTIVE
    assert record.governing_component == "master_equation"
    assert record.action_verdict == "EXECUTE"
    assert record.state_dim == 3
    assert record.subsystem_health is not None
    assert record.subsystem_health["risk_engine"] == SubsystemStatus.HEALTHY

    # Verify immutability (frozen dataclass)
    with pytest.raises(AttributeError):
        record.certeza = 0.5  # type: ignore[misc]


def test_decision_trace_to_dict_backward_compatibility() -> None:
    """Verifies that to_dict exports all ISO fields and legacy keys without breakage."""
    record = DecisionTraceRecord.create(
        delta_state=np.array([0.1]),
        delta_time=0.5,
        phi_moe_base=0.6,
        i_cvar=0.0,
        lambda_t_crono=0.8,
        certeza=0.45,
        magnitud_objetivo=0.005,
        momentum_veto=0.0,
        shadow_mode=True,
        action_verdict="HOLD",
        variable_destino=0.32,
    )

    trace = record.to_dict()

    # Core ISO keys
    assert trace["iso_standard"] == "ISO/IEC 22989:2022 §5.3"
    assert trace["execution_mode"] == "shadow"
    assert trace["governing_component"] == "phi_moe_base"

    # Legacy expected keys
    assert "telemetry_hash" in trace
    assert trace["phi_moe_base"] == 0.6
    assert trace["I_cvar"] == 0.0
    assert trace["lambda_t_crono"] == 0.8
    assert trace["certeza"] == 0.45
    assert trace["phi_redrose"] == 0.45
    assert trace["variable_destino"] == 0.32
    assert trace["D_t"] == 0.32
