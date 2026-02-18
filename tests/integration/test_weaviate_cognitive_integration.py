"""Integration tests for WeaviateCognitiveAdapter against a live Weaviate instance.

Guarded by env var ``WEAVIATE_INTEGRATION_TEST=1``.
Skipped automatically if Weaviate is not reachable.

Usage:
    WEAVIATE_INTEGRATION_TEST=1 pytest iot_machine_learning/tests/integration/test_weaviate_cognitive_integration.py -v
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

from iot_machine_learning.domain.entities.anomaly import (
    AnomalyResult,
    AnomalySeverity,
)
from iot_machine_learning.domain.entities.memory_search_result import (
    MemorySearchResult,
)
from iot_machine_learning.domain.entities.pattern import (
    PatternResult,
    PatternType,
)
from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.infrastructure.adapters.weaviate import (
    WeaviateCognitiveAdapter,
)

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")

_skip_reason = "Set WEAVIATE_INTEGRATION_TEST=1 to run integration tests"
_integration_enabled = os.environ.get("WEAVIATE_INTEGRATION_TEST", "0") == "1"


def _weaviate_reachable() -> bool:
    """Check if Weaviate is reachable."""
    try:
        req = urllib.request.Request(
            f"{WEAVIATE_URL}/v1/.well-known/ready", method="GET"
        )
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _integration_enabled, reason=_skip_reason),
    pytest.mark.skipif(
        _integration_enabled and not _weaviate_reachable(),
        reason=f"Weaviate not reachable at {WEAVIATE_URL}",
    ),
]


@pytest.fixture
def adapter() -> WeaviateCognitiveAdapter:
    return WeaviateCognitiveAdapter(
        base_url=WEAVIATE_URL,
        enabled=True,
        dry_run=False,
    )


@pytest.fixture
def prediction() -> Prediction:
    return Prediction(
        series_id="integration-test-42",
        predicted_value=25.5,
        confidence_score=0.85,
        trend="up",
        engine_name="taylor",
        horizon_steps=1,
        feature_contributions={"lag_1": 0.6},
        metadata={"explanation": "Integration test: upward trend with high confidence"},
        audit_trace_id="integration-trace-001",
    )


@pytest.fixture
def anomaly() -> AnomalyResult:
    return AnomalyResult(
        series_id="integration-test-42",
        is_anomaly=True,
        score=0.87,
        method_votes={"isolation_forest": 0.9, "z_score": 0.8},
        confidence=0.88,
        explanation="Integration test: sudden variance increase after stability",
        severity=AnomalySeverity.HIGH,
        context={"regime": "active"},
        audit_trace_id="integration-trace-002",
    )


@pytest.fixture
def pattern() -> PatternResult:
    return PatternResult(
        series_id="integration-test-42",
        pattern_type=PatternType.DRIFTING,
        confidence=0.75,
        description="Integration test: gradual upward drift over 2h window",
        metadata={"slope": 0.003},
    )


@pytest.fixture
def decision() -> dict:
    return {
        "device_id": 99,
        "pattern_signature": "integration-sig-001",
        "decision_type": "investigate",
        "priority": "high",
        "severity": "warning",
        "title": "Integration test decision",
        "summary": "Integration test: correlated drift across sensors",
        "explanation": "Integration test: sensors 42, 43 show upward drift",
        "recommended_actions": ["inspect_device"],
        "affected_series_ids": ["integration-test-42", "integration-test-43"],
        "event_count": 3,
        "confidence_score": 0.82,
        "is_recurring": False,
        "historical_resolution_rate": 0.0,
        "reason_trace": {"rules": ["correlated_drift"]},
        "audit_trace_id": "integration-trace-003",
    }


class TestRememberExplanationIntegration:
    def test_creates_object_in_weaviate(self, adapter, prediction):
        uuid = adapter.remember_explanation(
            prediction,
            source_record_id=99999,
            explanation_text="Integration test: upward trend with high confidence",
            domain_name="integration-test",
        )
        assert uuid is not None
        assert len(uuid) == 36  # UUID format

    def test_recall_finds_created_explanation(self, adapter, prediction):
        adapter.remember_explanation(
            prediction,
            source_record_id=99998,
            explanation_text="Integration test: unique recall test explanation alpha",
            domain_name="integration-test",
        )

        results = adapter.recall_similar_explanations(
            "unique recall test explanation alpha",
            limit=3,
            min_certainty=0.5,
        )
        assert len(results) >= 1
        assert any("alpha" in r.text.lower() for r in results)
        assert all(isinstance(r, MemorySearchResult) for r in results)


class TestRememberAnomalyIntegration:
    def test_creates_object_in_weaviate(self, adapter, anomaly):
        uuid = adapter.remember_anomaly(
            anomaly,
            source_record_id=99997,
            event_code="ANOMALY_DETECTED",
            behavior_pattern="spike",
            domain_name="integration-test",
        )
        assert uuid is not None
        assert len(uuid) == 36


class TestRememberPatternIntegration:
    def test_creates_object_in_weaviate(self, adapter, pattern):
        uuid = adapter.remember_pattern(
            pattern,
            source_record_id=99996,
            domain_name="integration-test",
        )
        assert uuid is not None
        assert len(uuid) == 36


class TestRememberDecisionIntegration:
    def test_creates_object_in_weaviate(self, adapter, decision):
        uuid = adapter.remember_decision(
            decision,
            source_record_id=99995,
            domain_name="integration-test",
        )
        assert uuid is not None
        assert len(uuid) == 36
