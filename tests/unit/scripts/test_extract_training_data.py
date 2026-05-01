"""Tests for training_extractor.py — no DB required."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from scripts.training_extractor import (
    _extract_fields,
    _is_valid_vector,
    _parse_ml_result,
    build_training_records,
    compute_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_VECTOR: List[float] = [float(i) / 18.0 for i in range(18)]

_SAMPLE_ML_RESULT = {
    "analysis": {
        "domain": "security",
        "situation_vector": _VALID_VECTOR,
        "urgency_score": 0.85,
        "adaptive_thresholds": {"current_severity": "high"},
    },
    "confidence": 0.72,
}


def _make_row(
    ml_result: Dict[str, Any] = _SAMPLE_ML_RESULT,
    conclusion: str = "Security incident detected.",
    analysis_id: str = "aaaaaaaa-0000-0000-0000-000000000001",
) -> Dict[str, Any]:
    return {
        "full_text": "intentos fallidos autenticación",
        "ml_result_json": json.dumps(ml_result),
        "conclusion": conclusion,
        "analysis_id": analysis_id,
    }


# ---------------------------------------------------------------------------
# _parse_ml_result
# ---------------------------------------------------------------------------

class TestParseMlResult:
    def test_valid_json(self):
        result = _parse_ml_result(json.dumps({"a": 1}))
        assert result == {"a": 1}

    def test_none_returns_empty(self):
        assert _parse_ml_result(None) == {}

    def test_empty_string_returns_empty(self):
        assert _parse_ml_result("") == {}

    def test_invalid_json_returns_empty(self):
        assert _parse_ml_result("{not valid}") == {}


# ---------------------------------------------------------------------------
# _extract_fields
# ---------------------------------------------------------------------------

class TestExtractFields:
    def test_extracts_all_fields(self):
        fields = _extract_fields(_SAMPLE_ML_RESULT)
        assert fields["domain"] == "security"
        assert fields["situation_vector"] == _VALID_VECTOR
        assert fields["severity"] == "high"
        assert fields["urgency_score"] == pytest.approx(0.85)
        assert fields["confidence"] == pytest.approx(0.72)

    def test_defaults_on_empty_blob(self):
        fields = _extract_fields({})
        assert fields["domain"] == "general"
        assert fields["situation_vector"] == []
        assert fields["severity"] == "info"
        assert fields["urgency_score"] == 0.0
        assert fields["confidence"] == 0.0

    def test_severity_nested_path(self):
        blob = {"analysis": {"adaptive_thresholds": {"current_severity": "critical"}}}
        assert _extract_fields(blob)["severity"] == "critical"


# ---------------------------------------------------------------------------
# _is_valid_vector
# ---------------------------------------------------------------------------

class TestIsValidVector:
    def test_valid_18_dim(self):
        assert _is_valid_vector(_VALID_VECTOR) is True

    def test_empty_list_invalid(self):
        assert _is_valid_vector([]) is False

    def test_17_dim_invalid(self):
        assert _is_valid_vector([0.0] * 17) is False

    def test_19_dim_invalid(self):
        assert _is_valid_vector([0.0] * 19) is False

    def test_non_list_invalid(self):
        assert _is_valid_vector(None) is False
        assert _is_valid_vector("vector") is False


# ---------------------------------------------------------------------------
# build_training_records
# ---------------------------------------------------------------------------

class TestBuildTrainingRecords:
    def test_valid_row_produces_record(self):
        records = build_training_records([_make_row()])
        assert len(records) == 1
        r = records[0]
        assert r["domain"] == "security"
        assert len(r["situation_vector"]) == 18
        assert r["severity"] == "high"
        assert r["conclusion"] == "Security incident detected."

    def test_missing_vector_is_filtered(self):
        blob = dict(_SAMPLE_ML_RESULT)
        blob["analysis"] = {**_SAMPLE_ML_RESULT["analysis"], "situation_vector": []}
        records = build_training_records([_make_row(ml_result=blob)])
        assert records == []

    def test_short_vector_is_filtered(self):
        blob = dict(_SAMPLE_ML_RESULT)
        blob["analysis"] = {**_SAMPLE_ML_RESULT["analysis"], "situation_vector": [0.1] * 5}
        records = build_training_records([_make_row(ml_result=blob)])
        assert records == []

    def test_multiple_rows_filtered_correctly(self):
        bad_blob = {**_SAMPLE_ML_RESULT, "analysis": {**_SAMPLE_ML_RESULT["analysis"], "situation_vector": []}}
        rows = [_make_row(), _make_row(ml_result=bad_blob), _make_row()]
        records = build_training_records(rows)
        assert len(records) == 2

    def test_record_has_all_required_keys(self):
        records = build_training_records([_make_row()])
        required = {"analysis_id", "situation_vector", "domain", "severity",
                    "urgency_score", "confidence", "conclusion"}
        assert required == set(records[0].keys())

    def test_output_is_json_serializable(self):
        records = build_training_records([_make_row()])
        dumped = json.dumps(records)
        loaded = json.loads(dumped)
        assert loaded[0]["domain"] == "security"

    def test_empty_input_returns_empty(self):
        assert build_training_records([]) == []


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------

class TestComputeSummary:
    def _records(self):
        return build_training_records([
            _make_row(
                ml_result={**_SAMPLE_ML_RESULT,
                           "analysis": {**_SAMPLE_ML_RESULT["analysis"], "domain": "security"}},
                analysis_id="id-1",
            ),
            _make_row(
                ml_result={**_SAMPLE_ML_RESULT,
                           "analysis": {**_SAMPLE_ML_RESULT["analysis"], "domain": "infrastructure",
                                        "adaptive_thresholds": {"current_severity": "warning"}}},
                analysis_id="id-2",
            ),
            _make_row(
                ml_result={**_SAMPLE_ML_RESULT,
                           "analysis": {**_SAMPLE_ML_RESULT["analysis"], "domain": "security"}},
                analysis_id="id-3",
            ),
        ])

    def test_domain_counts(self):
        summary = compute_summary(self._records())
        assert summary["by_domain"]["security"] == 2
        assert summary["by_domain"]["infrastructure"] == 1

    def test_severity_counts(self):
        summary = compute_summary(self._records())
        assert summary["by_severity"]["high"] == 2
        assert summary["by_severity"]["warning"] == 1

    def test_empty_records(self):
        summary = compute_summary([])
        assert summary == {"by_domain": {}, "by_severity": {}}
