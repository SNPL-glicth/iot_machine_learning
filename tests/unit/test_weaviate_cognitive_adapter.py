"""Unit tests for WeaviateCognitiveAdapter.

All HTTP calls are mocked — no Weaviate instance required.
Tests verify:
    - Correct payload construction for all 4 remember_* methods
    - Correct GraphQL query construction for all 4 recall_* methods
    - Fail-safe behaviour (errors → None/[] instead of exceptions)
    - Dry-run mode (logs but does not send)
    - Disabled mode (returns immediately)
    - Feature flag integration
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Dict
from unittest.mock import MagicMock, patch

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
from iot_machine_learning.domain.ports.cognitive_memory_port import (
    CognitiveMemoryPort,
)
from iot_machine_learning.infrastructure.adapters.weaviate_cognitive import (
    WeaviateCognitiveAdapter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter() -> WeaviateCognitiveAdapter:
    return WeaviateCognitiveAdapter(
        base_url="http://fake-weaviate:8080",
        enabled=True,
        dry_run=False,
    )


@pytest.fixture
def dry_run_adapter() -> WeaviateCognitiveAdapter:
    return WeaviateCognitiveAdapter(
        base_url="http://fake-weaviate:8080",
        enabled=True,
        dry_run=True,
    )


@pytest.fixture
def disabled_adapter() -> WeaviateCognitiveAdapter:
    return WeaviateCognitiveAdapter(
        base_url="http://fake-weaviate:8080",
        enabled=False,
    )


@pytest.fixture
def sample_prediction() -> Prediction:
    return Prediction(
        series_id="42",
        predicted_value=25.5,
        confidence_score=0.85,
        trend="up",
        engine_name="taylor",
        horizon_steps=1,
        confidence_interval=(24.0, 27.0),
        feature_contributions={"lag_1": 0.6, "trend": 0.4},
        metadata={"explanation": "Upward trend detected with high confidence"},
        audit_trace_id="trace-abc-123",
    )


@pytest.fixture
def sample_anomaly() -> AnomalyResult:
    return AnomalyResult(
        series_id="42",
        is_anomaly=True,
        score=0.87,
        method_votes={"isolation_forest": 0.9, "z_score": 0.8, "iqr": 0.85},
        confidence=0.88,
        explanation="Sudden variance increase after prolonged stability",
        severity=AnomalySeverity.HIGH,
        context={"regime": "active", "correlation": 0.92},
        audit_trace_id="trace-def-456",
    )


@pytest.fixture
def sample_pattern() -> PatternResult:
    return PatternResult(
        series_id="42",
        pattern_type=PatternType.DRIFTING,
        confidence=0.75,
        description="Gradual upward drift detected over 2h window",
        metadata={"slope": 0.003, "r_squared": 0.91},
    )


@pytest.fixture
def sample_decision() -> Dict[str, Any]:
    return {
        "device_id": 7,
        "pattern_signature": "sig-abc-123",
        "decision_type": "investigate",
        "priority": "high",
        "severity": "warning",
        "title": "Multiple sensors drifting",
        "summary": "Device 7 shows correlated drift across 3 sensors",
        "explanation": "Sensors 42, 43, 44 all show upward drift",
        "recommended_actions": ["inspect_device", "check_calibration"],
        "affected_series_ids": ["42", "43", "44"],
        "event_count": 5,
        "confidence_score": 0.82,
        "is_recurring": True,
        "historical_resolution_rate": 0.7,
        "reason_trace": {"rules": ["correlated_drift"]},
        "audit_trace_id": "trace-ghi-789",
    }


def _mock_urlopen_response(data: Dict[str, Any]) -> MagicMock:
    """Create a mock urllib response with JSON body."""
    body = json.dumps(data).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Test: isinstance check
# ---------------------------------------------------------------------------

class TestAdapterContract:
    def test_implements_cognitive_memory_port(self, adapter):
        assert isinstance(adapter, CognitiveMemoryPort)


# ---------------------------------------------------------------------------
# Test: disabled mode
# ---------------------------------------------------------------------------

class TestDisabledMode:
    def test_remember_explanation_returns_none(
        self, disabled_adapter, sample_prediction
    ):
        result = disabled_adapter.remember_explanation(
            sample_prediction, 1, explanation_text="test"
        )
        assert result is None

    def test_remember_anomaly_returns_none(
        self, disabled_adapter, sample_anomaly
    ):
        result = disabled_adapter.remember_anomaly(sample_anomaly, 1)
        assert result is None

    def test_remember_pattern_returns_none(
        self, disabled_adapter, sample_pattern
    ):
        result = disabled_adapter.remember_pattern(sample_pattern)
        assert result is None

    def test_remember_decision_returns_none(
        self, disabled_adapter, sample_decision
    ):
        result = disabled_adapter.remember_decision(sample_decision, 1)
        assert result is None

    def test_recall_explanations_returns_empty(self, disabled_adapter):
        result = disabled_adapter.recall_similar_explanations("test")
        assert result == []

    def test_recall_anomalies_returns_empty(self, disabled_adapter):
        result = disabled_adapter.recall_similar_anomalies("test")
        assert result == []

    def test_recall_patterns_returns_empty(self, disabled_adapter):
        result = disabled_adapter.recall_similar_patterns("test")
        assert result == []

    def test_recall_decisions_returns_empty(self, disabled_adapter):
        result = disabled_adapter.recall_similar_decisions("test")
        assert result == []


# ---------------------------------------------------------------------------
# Test: dry-run mode
# ---------------------------------------------------------------------------

class TestDryRunMode:
    def test_remember_explanation_returns_dry_run_uuid(
        self, dry_run_adapter, sample_prediction
    ):
        result = dry_run_adapter.remember_explanation(
            sample_prediction, 100, explanation_text="test explanation"
        )
        assert result == "dry-run-uuid"

    def test_remember_anomaly_returns_dry_run_uuid(
        self, dry_run_adapter, sample_anomaly
    ):
        result = dry_run_adapter.remember_anomaly(sample_anomaly, 200)
        assert result == "dry-run-uuid"

    def test_remember_pattern_returns_dry_run_uuid(
        self, dry_run_adapter, sample_pattern
    ):
        result = dry_run_adapter.remember_pattern(sample_pattern)
        assert result == "dry-run-uuid"

    def test_remember_decision_returns_dry_run_uuid(
        self, dry_run_adapter, sample_decision
    ):
        result = dry_run_adapter.remember_decision(sample_decision, 300)
        assert result == "dry-run-uuid"

    def test_recall_explanations_returns_empty_in_dry_run(
        self, dry_run_adapter
    ):
        result = dry_run_adapter.recall_similar_explanations("test")
        assert result == []


# ---------------------------------------------------------------------------
# Test: remember_explanation (mocked HTTP)
# ---------------------------------------------------------------------------

class TestRememberExplanation:
    @patch("urllib.request.urlopen")
    def test_creates_object_and_returns_uuid(
        self, mock_urlopen, adapter, sample_prediction
    ):
        mock_urlopen.return_value = _mock_urlopen_response(
            {"id": "uuid-explanation-1", "class": "MLExplanation"}
        )

        result = adapter.remember_explanation(
            sample_prediction,
            100,
            explanation_text="Upward trend with high confidence",
        )

        assert result == "uuid-explanation-1"
        mock_urlopen.assert_called_once()

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))

        assert body["class"] == "MLExplanation"
        assert body["properties"]["seriesId"] == "42"
        assert body["properties"]["engineName"] == "taylor"
        assert body["properties"]["trend"] == "up"
        assert body["properties"]["confidenceScore"] == 0.85
        assert body["properties"]["confidenceLevel"] == "high"
        assert body["properties"]["predictedValue"] == 25.5
        assert body["properties"]["sourceRecordId"] == 100
        assert body["properties"]["auditTraceId"] == "trace-abc-123"
        assert body["properties"]["explanationText"] == "Upward trend with high confidence"

    @patch("urllib.request.urlopen")
    def test_uses_metadata_explanation_when_text_empty(
        self, mock_urlopen, adapter, sample_prediction
    ):
        mock_urlopen.return_value = _mock_urlopen_response(
            {"id": "uuid-2", "class": "MLExplanation"}
        )

        adapter.remember_explanation(sample_prediction, 101)

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))
        assert body["properties"]["explanationText"] == "Upward trend detected with high confidence"

    @patch("urllib.request.urlopen")
    def test_returns_none_on_http_error(
        self, mock_urlopen, adapter, sample_prediction
    ):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://fake", code=500, msg="Server Error",
            hdrs=None, fp=BytesIO(b"internal error"),
        )

        result = adapter.remember_explanation(
            sample_prediction, 100, explanation_text="test"
        )
        assert result is None

    @patch("urllib.request.urlopen")
    def test_returns_none_on_connection_error(
        self, mock_urlopen, adapter, sample_prediction
    ):
        mock_urlopen.side_effect = ConnectionError("refused")

        result = adapter.remember_explanation(
            sample_prediction, 100, explanation_text="test"
        )
        assert result is None


# ---------------------------------------------------------------------------
# Test: remember_anomaly (mocked HTTP)
# ---------------------------------------------------------------------------

class TestRememberAnomaly:
    @patch("urllib.request.urlopen")
    def test_creates_object_and_returns_uuid(
        self, mock_urlopen, adapter, sample_anomaly
    ):
        mock_urlopen.return_value = _mock_urlopen_response(
            {"id": "uuid-anomaly-1", "class": "AnomalyMemory"}
        )

        result = adapter.remember_anomaly(
            sample_anomaly, 200,
            event_code="ANOMALY_DETECTED",
            behavior_pattern="drifting",
        )

        assert result == "uuid-anomaly-1"

        request = mock_urlopen.call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))

        assert body["class"] == "AnomalyMemory"
        assert body["properties"]["seriesId"] == "42"
        assert body["properties"]["isAnomaly"] is True
        assert body["properties"]["anomalyScore"] == 0.87
        assert body["properties"]["severity"] == "high"
        assert body["properties"]["eventCode"] == "ANOMALY_DETECTED"
        assert body["properties"]["behaviorPattern"] == "drifting"
        assert "Sudden variance" in body["properties"]["explanationText"]


# ---------------------------------------------------------------------------
# Test: remember_pattern (mocked HTTP)
# ---------------------------------------------------------------------------

class TestRememberPattern:
    @patch("urllib.request.urlopen")
    def test_creates_object_and_returns_uuid(
        self, mock_urlopen, adapter, sample_pattern
    ):
        mock_urlopen.return_value = _mock_urlopen_response(
            {"id": "uuid-pattern-1", "class": "PatternMemory"}
        )

        result = adapter.remember_pattern(sample_pattern, source_record_id=50)

        assert result == "uuid-pattern-1"

        request = mock_urlopen.call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))

        assert body["class"] == "PatternMemory"
        assert body["properties"]["seriesId"] == "42"
        assert body["properties"]["patternType"] == "drifting"
        assert body["properties"]["confidence"] == 0.75
        assert "Gradual upward drift" in body["properties"]["descriptionText"]
        assert body["properties"]["sourceRecordId"] == 50


# ---------------------------------------------------------------------------
# Test: remember_decision (mocked HTTP)
# ---------------------------------------------------------------------------

class TestRememberDecision:
    @patch("urllib.request.urlopen")
    def test_creates_object_and_returns_uuid(
        self, mock_urlopen, adapter, sample_decision
    ):
        mock_urlopen.return_value = _mock_urlopen_response(
            {"id": "uuid-decision-1", "class": "DecisionReasoning"}
        )

        result = adapter.remember_decision(sample_decision, 300)

        assert result == "uuid-decision-1"

        request = mock_urlopen.call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))

        assert body["class"] == "DecisionReasoning"
        assert body["properties"]["deviceId"] == 7
        assert body["properties"]["decisionType"] == "investigate"
        assert body["properties"]["priority"] == "high"
        assert body["properties"]["affectedSeriesIds"] == ["42", "43", "44"]
        assert body["properties"]["eventCount"] == 5
        assert body["properties"]["isRecurring"] is True
        assert body["properties"]["sourceRecordId"] == 300


# ---------------------------------------------------------------------------
# Test: recall_similar_explanations (mocked HTTP)
# ---------------------------------------------------------------------------

class TestRecallExplanations:
    @patch("urllib.request.urlopen")
    def test_returns_memory_search_results(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _mock_urlopen_response({
            "data": {
                "Get": {
                    "MLExplanation": [
                        {
                            "seriesId": "42",
                            "explanationText": "Upward trend detected",
                            "confidenceScore": 0.85,
                            "sourceRecordId": 100,
                            "createdAt": "2026-02-12T15:00:00Z",
                            "engineName": "taylor",
                            "trend": "up",
                            "_additional": {
                                "id": "uuid-1",
                                "certainty": 0.92,
                            },
                        },
                    ],
                },
            },
        })

        results = adapter.recall_similar_explanations(
            "upward trend", series_id="42", limit=3
        )

        assert len(results) == 1
        r = results[0]
        assert isinstance(r, MemorySearchResult)
        assert r.memory_id == "uuid-1"
        assert r.series_id == "42"
        assert r.text == "Upward trend detected"
        assert r.certainty == 0.92
        assert r.source_record_id == 100
        assert r.created_at == "2026-02-12T15:00:00Z"
        assert r.is_high_certainty is True

    @patch("urllib.request.urlopen")
    def test_returns_empty_on_no_results(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _mock_urlopen_response({
            "data": {"Get": {"MLExplanation": []}},
        })

        results = adapter.recall_similar_explanations("nonexistent")
        assert results == []

    @patch("urllib.request.urlopen")
    def test_returns_empty_on_error(self, mock_urlopen, adapter):
        mock_urlopen.side_effect = ConnectionError("refused")

        results = adapter.recall_similar_explanations("test")
        assert results == []

    @patch("urllib.request.urlopen")
    def test_graphql_query_includes_where_filter(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _mock_urlopen_response({
            "data": {"Get": {"MLExplanation": []}},
        })

        adapter.recall_similar_explanations(
            "test", series_id="42", engine_name="taylor"
        )

        request = mock_urlopen.call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))
        query = body["query"]
        assert "where:" in query
        assert "seriesId" in query
        assert "engineName" in query


# ---------------------------------------------------------------------------
# Test: recall_similar_anomalies (mocked HTTP)
# ---------------------------------------------------------------------------

class TestRecallAnomalies:
    @patch("urllib.request.urlopen")
    def test_returns_memory_search_results(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _mock_urlopen_response({
            "data": {
                "Get": {
                    "AnomalyMemory": [
                        {
                            "seriesId": "42",
                            "explanationText": "Sudden spike detected",
                            "anomalyScore": 0.9,
                            "severity": "high",
                            "sourceRecordId": 200,
                            "createdAt": "2026-02-12T15:00:00Z",
                            "eventCode": "ANOMALY_DETECTED",
                            "behaviorPattern": "spike",
                            "_additional": {"id": "uuid-a1", "certainty": 0.88},
                        },
                    ],
                },
            },
        })

        results = adapter.recall_similar_anomalies("spike", severity="high")
        assert len(results) == 1
        assert results[0].series_id == "42"
        assert results[0].certainty == 0.88


# ---------------------------------------------------------------------------
# Test: recall_similar_patterns (mocked HTTP)
# ---------------------------------------------------------------------------

class TestRecallPatterns:
    @patch("urllib.request.urlopen")
    def test_returns_memory_search_results(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _mock_urlopen_response({
            "data": {
                "Get": {
                    "PatternMemory": [
                        {
                            "seriesId": "42",
                            "descriptionText": "Oscillating pattern",
                            "patternType": "oscillating",
                            "confidence": 0.8,
                            "sourceRecordId": 50,
                            "createdAt": "2026-02-12T15:00:00Z",
                            "_additional": {"id": "uuid-p1", "certainty": 0.85},
                        },
                    ],
                },
            },
        })

        results = adapter.recall_similar_patterns("oscillating")
        assert len(results) == 1
        assert results[0].text == "Oscillating pattern"


# ---------------------------------------------------------------------------
# Test: recall_similar_decisions (mocked HTTP)
# ---------------------------------------------------------------------------

class TestRecallDecisions:
    @patch("urllib.request.urlopen")
    def test_returns_memory_search_results(self, mock_urlopen, adapter):
        mock_urlopen.return_value = _mock_urlopen_response({
            "data": {
                "Get": {
                    "DecisionReasoning": [
                        {
                            "summaryText": "Correlated drift on device 7",
                            "explanationText": "Full reasoning chain",
                            "severity": "warning",
                            "decisionType": "investigate",
                            "sourceRecordId": 300,
                            "createdAt": "2026-02-12T15:00:00Z",
                            "affectedSeriesIds": ["42", "43"],
                            "_additional": {"id": "uuid-d1", "certainty": 0.80},
                        },
                    ],
                },
            },
        })

        results = adapter.recall_similar_decisions("correlated drift")
        assert len(results) == 1
        r = results[0]
        assert r.text == "Correlated drift on device 7"
        assert r.series_id == "42"
        assert r.source_record_id == 300


# ---------------------------------------------------------------------------
# Test: fail-safe behaviour
# ---------------------------------------------------------------------------

class TestFailSafe:
    @patch("urllib.request.urlopen")
    def test_remember_explanation_never_raises(
        self, mock_urlopen, adapter, sample_prediction
    ):
        mock_urlopen.side_effect = Exception("catastrophic failure")
        result = adapter.remember_explanation(
            sample_prediction, 1, explanation_text="test"
        )
        assert result is None

    @patch("urllib.request.urlopen")
    def test_remember_anomaly_never_raises(
        self, mock_urlopen, adapter, sample_anomaly
    ):
        mock_urlopen.side_effect = Exception("catastrophic failure")
        result = adapter.remember_anomaly(sample_anomaly, 1)
        assert result is None

    @patch("urllib.request.urlopen")
    def test_remember_pattern_never_raises(
        self, mock_urlopen, adapter, sample_pattern
    ):
        mock_urlopen.side_effect = Exception("catastrophic failure")
        result = adapter.remember_pattern(sample_pattern)
        assert result is None

    @patch("urllib.request.urlopen")
    def test_remember_decision_never_raises(
        self, mock_urlopen, adapter, sample_decision
    ):
        mock_urlopen.side_effect = Exception("catastrophic failure")
        result = adapter.remember_decision(sample_decision, 1)
        assert result is None

    @patch("urllib.request.urlopen")
    def test_recall_explanations_never_raises(self, mock_urlopen, adapter):
        mock_urlopen.side_effect = Exception("catastrophic failure")
        result = adapter.recall_similar_explanations("test")
        assert result == []

    @patch("urllib.request.urlopen")
    def test_recall_anomalies_never_raises(self, mock_urlopen, adapter):
        mock_urlopen.side_effect = Exception("catastrophic failure")
        result = adapter.recall_similar_anomalies("test")
        assert result == []

    @patch("urllib.request.urlopen")
    def test_recall_patterns_never_raises(self, mock_urlopen, adapter):
        mock_urlopen.side_effect = Exception("catastrophic failure")
        result = adapter.recall_similar_patterns("test")
        assert result == []

    @patch("urllib.request.urlopen")
    def test_recall_decisions_never_raises(self, mock_urlopen, adapter):
        mock_urlopen.side_effect = Exception("catastrophic failure")
        result = adapter.recall_similar_decisions("test")
        assert result == []


# ---------------------------------------------------------------------------
# Test: where filter construction
# ---------------------------------------------------------------------------

class TestWhereFilterConstruction:
    def test_single_filter(self, adapter):
        f = adapter._build_where_filter([
            adapter._where_eq_text("seriesId", "42"),
        ])
        assert f == {
            "path": ["seriesId"],
            "operator": "Equal",
            "valueText": "42",
        }

    def test_multiple_filters_combined_with_and(self, adapter):
        f = adapter._build_where_filter([
            adapter._where_eq_text("seriesId", "42"),
            adapter._where_eq_text("severity", "high"),
        ])
        assert f["operator"] == "And"
        assert len(f["operands"]) == 2

    def test_none_values_are_skipped(self, adapter):
        f = adapter._build_where_filter([
            adapter._where_eq_text("seriesId", None),
            adapter._where_eq_text("severity", "high"),
        ])
        assert f == {
            "path": ["severity"],
            "operator": "Equal",
            "valueText": "high",
        }

    def test_all_none_returns_none(self, adapter):
        f = adapter._build_where_filter([
            adapter._where_eq_text("seriesId", None),
            adapter._where_eq_text("severity", None),
        ])
        assert f is None

    def test_int_filter(self, adapter):
        f = adapter._where_eq_int("deviceId", 7)
        assert f == {
            "path": ["deviceId"],
            "operator": "Equal",
            "valueInt": 7,
        }
