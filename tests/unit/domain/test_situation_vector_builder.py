"""Tests for SituationVectorBuilder — Step 4 numeric pipeline."""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.entities.explainability.explanation import (
    Explanation,
    Outcome,
)
from iot_machine_learning.domain.entities.explainability.reasoning_trace import (
    ReasoningTrace,
)
from iot_machine_learning.domain.entities.explainability.signal_snapshot import (
    SignalSnapshot,
)
from iot_machine_learning.domain.services.situation_vector_builder import (
    _SITUATION_VECTOR_DIM,
    _clamp01,
    _circuit_to_numeric,
    build_situation_vector,
)


class TestClampAndHelpers:
    def test_clamp01(self):
        assert _clamp01(-0.5) == 0.0
        assert _clamp01(0.5) == 0.5
        assert _clamp01(1.5) == 1.0

    def test_circuit_numeric(self):
        assert _circuit_to_numeric("closed") == 0.0
        assert _circuit_to_numeric("half_open") == 0.5
        assert _circuit_to_numeric("open") == 1.0
        assert _circuit_to_numeric("UNKNOWN") == 0.0


class TestBuildSituationVector:
    def test_none_explanation_returns_zeros(self):
        vec = build_situation_vector(None)
        assert len(vec) == 18
        assert all(v == 0.0 for v in vec)

    def test_minimal_explanation_fills_18_dims(self):
        explanation = Explanation(
            series_id="test",
            signal=SignalSnapshot(
                regime_vector=[1.0, 2.0, 0.5, 0.1, 0.2, 0.3],
            ),
            outcome=Outcome(
                confidence=0.85,
                anomaly_score=0.4,
                extra={"composite_score": 0.72},
            ),
            trace=ReasoningTrace(
                n_engines_available=4,
                n_engines_active=2,
            ),
        )
        vec = build_situation_vector(explanation)
        assert len(vec) == 18
        # regime dims should be non-zero (soft-clamped)
        assert vec[0] != 0.0
        assert vec[1] != 0.0
        assert vec[6] == pytest.approx(0.85, abs=1e-6)
        assert vec[7] == pytest.approx(0.4, abs=1e-6)
        assert vec[8] == pytest.approx(0.72, abs=1e-6)
        # n_engines_ratio = 2/4 = 0.5
        assert vec[17] == pytest.approx(0.5, abs=1e-6)

    def test_moe_vector_injected(self):
        explanation = Explanation(series_id="test")
        metadata = {
            "moe_vector": [0.6, 0.3, 0.1, 0.8, 0.2],
            "cognitive_trace": {
                "drift_score": 2.5,
                "circuit_breaker_status": "half_open",
                "amnesic_mode": True,
            },
        }
        vec = build_situation_vector(explanation, metadata=metadata)
        # 18 = 13 dims activas + 5 reservadas MoE (dims 12-16, siempre 0.0)
        assert len(vec) == _SITUATION_VECTOR_DIM
        assert vec[9] == pytest.approx(0.5, abs=1e-6)   # 2.5/5
        assert vec[10] == pytest.approx(0.5, abs=1e-6)  # half_open
        assert vec[11] == 1.0                           # amnesic
        assert vec[12] == pytest.approx(0.6, abs=1e-6)
        assert vec[13] == pytest.approx(0.3, abs=1e-6)
        assert vec[14] == pytest.approx(0.1, abs=1e-6)
        assert vec[15] == pytest.approx(0.8, abs=1e-6)
        assert vec[16] == pytest.approx(0.2, abs=1e-6)

    def test_all_values_clamped_to_01(self):
        explanation = Explanation(
            series_id="test",
            signal=SignalSnapshot(regime_vector=[100.0, -100.0, 0.0, 0.0, 0.0, 0.0]),
            outcome=Outcome(
                confidence=0.9,
                anomaly_score=0.3,
                extra={"composite_score": 0.95},
            ),
        )
        vec = build_situation_vector(explanation)
        assert all(0.0 <= v <= 1.0 for v in vec)
