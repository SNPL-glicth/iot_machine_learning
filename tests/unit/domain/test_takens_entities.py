"""Unit tests for Phase 1 Takens Domain Entities."""

from __future__ import annotations

import numpy as np
import pytest

from domain.entities.takens import (
    TakensParameters,
    EmbeddedState,
    TopologicalAuditRecord,
)


def test_takens_parameters_defaults_and_validation():
    params = TakensParameters()
    assert params.m == 4
    assert params.buffer_capacity == 1024
    assert params.tau_strides == (1, 2, 4, 8)
    assert params.tau_fnn == 0.40

    # Inmutability (frozen)
    with pytest.raises(Exception):
        params.m = 6

    # Validation: non-power-of-two buffer
    with pytest.raises(ValueError, match="positive power of 2"):
        TakensParameters(buffer_capacity=1000)

    # Validation: m < 2
    with pytest.raises(ValueError, match="dimension m must be >= 2"):
        TakensParameters(m=1)

    # Validation: invalid fnn threshold
    with pytest.raises(ValueError, match="tau_fnn threshold"):
        TakensParameters(tau_fnn=-0.1)


def test_embedded_state_sanitization_and_immutability():
    # Vector with NaNs and Infs
    raw_vec = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float64)
    state = EmbeddedState(
        delay_vector=raw_vec,
        effective_dimension=np.nan,
        fnn_ratio=1.5,  # Exceeds 1.0
        timestamp_ns=1700000000000,
    )

    # NaN / Inf replaced by finite numbers
    assert np.all(np.isfinite(state.delay_vector))
    assert state.delay_vector[1] == 0.0
    assert state.effective_dimension == 1.0  # NaN clamped to 1.0
    assert state.fnn_ratio == 1.0           # 1.5 clamped to 1.0
    assert state.embedding_dimension == 4
    assert state.metric_energy > 0.0

    # Strict array immutability check
    with pytest.raises(ValueError):
        state.delay_vector[0] = 999.0

    # to_dict check
    data = state.to_dict()
    assert isinstance(data["delay_vector"], list)
    assert data["embedding_dimension"] == 4


def test_topological_audit_record_trace():
    audit = TopologicalAuditRecord(
        timestamp_ns=1700000000123,
        d_effective=2.456789,
        fnn_ratio=0.123456,
        is_manifold_veto=True,
        veto_reason="fnn_threshold_exceeded",
        manifold_coherence=0.85,
        embedded_norm=4.2,
    )

    trace = audit.to_telemetry_trace()
    assert trace["timestamp_ns"] == 1700000000123
    assert trace["d_effective"] == 2.456789
    assert trace["is_manifold_veto"] is True
    assert trace["veto_reason"] == "fnn_threshold_exceeded"

    # NaN fault-tolerance
    corrupted_audit = TopologicalAuditRecord(
        timestamp_ns=-10,
        d_effective=float("nan"),
        fnn_ratio=float("nan"),
        is_manifold_veto=False,
        veto_reason="should_be_cleared",
        manifold_coherence=float("inf"),
        embedded_norm=float("-inf"),
    )
    assert corrupted_audit.timestamp_ns == 0
    assert corrupted_audit.d_effective == 1.0
    assert corrupted_audit.fnn_ratio == 0.0
    assert corrupted_audit.veto_reason is None
    assert corrupted_audit.manifold_coherence == 0.0
    assert corrupted_audit.embedded_norm == 0.0
