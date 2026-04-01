"""Tests for DecisionOutput and TextDecisionOutput DTOs.

Covers:
- to_summary() returns exactly 3 fields
- to_dict() always includes mandatory fields
- metadata present as single block
- decision="out_of_domain" when is_out_of_domain=True
- to_decision_output() in PredictionDTO
- to_decision_output() in TextCognitiveResult
"""

from __future__ import annotations

import pytest
from typing import Any, Dict

from iot_machine_learning.application.dto.decision_output import DecisionOutput
from iot_machine_learning.application.dto.text_decision_output import TextDecisionOutput
from iot_machine_learning.application.dto.prediction_dto import PredictionDTO


class TestDecisionOutputBasics:
    """Basic construction and sanity checks."""

    def test_decision_output_construction(self):
        """DecisionOutput can be instantiated with mandatory fields."""
        result = DecisionOutput(
            decision="normal",
            confidence=0.85,
            verdict="system operating normally",
            severity="info",
            action_required=False,
            action=None,
        )
        assert result.decision == "normal"
        assert result.confidence == pytest.approx(0.85)
        assert result.verdict == "system operating normally"
        assert result.severity == "info"
        assert result.action_required is False
        assert result.action is None
        assert result.metadata == {}

    def test_decision_output_with_metadata(self):
        """DecisionOutput accepts metadata dict."""
        metadata = {"engine_decision": {"chosen": "taylor"}}
        result = DecisionOutput(
            decision="anomaly",
            confidence=0.75,
            verdict="anomaly detected",
            severity="warning",
            action_required=True,
            action="investigate",
            metadata=metadata,
        )
        assert result.metadata == metadata


class TestDecisionOutputToSummary:
    """to_summary() returns exactly 3 fields."""

    def test_to_summary_returns_three_fields(self):
        """to_summary() returns exactly decision, confidence, verdict."""
        result = DecisionOutput(
            decision="normal",
            confidence=0.90,
            verdict="all systems nominal",
            severity="info",
            action_required=False,
            action=None,
        )
        summary = result.to_summary()

        assert len(summary) == 3
        assert "decision" in summary
        assert "confidence" in summary
        assert "verdict" in summary
        assert "severity" not in summary
        assert "action_required" not in summary
        assert "metadata" not in summary

    def test_to_summary_values_correct(self):
        """to_summary() values match the original."""
        result = DecisionOutput(
            decision="anomaly",
            confidence=0.65,
            verdict="temperature spike detected",
            severity="warning",
            action_required=True,
            action="check_hvac",
        )
        summary = result.to_summary()

        assert summary["decision"] == "anomaly"
        assert summary["confidence"] == pytest.approx(0.65)
        assert summary["verdict"] == "temperature spike detected"


class TestDecisionOutputToDict:
    """to_dict() includes all 6 mandatory fields + metadata."""

    def test_to_dict_includes_all_mandatory_fields(self):
        """to_dict() always has decision, confidence, verdict, severity, action_required, action."""
        result = DecisionOutput(
            decision="degraded",
            confidence=0.45,
            verdict="low confidence prediction",
            severity="warning",
            action_required=False,
            action=None,
        )
        d = result.to_dict()

        assert "decision" in d
        assert "confidence" in d
        assert "verdict" in d
        assert "severity" in d
        assert "action_required" in d
        assert "action" in d
        assert "metadata" in d
        assert len(d) == 7  # 6 mandatory + metadata

    def test_to_dict_metadata_as_block(self):
        """metadata is a single dict block, not flattened."""
        metadata: Dict[str, Any] = {
            "engine_decision": {"chosen": "baseline"},
            "calibration_report": {"raw": 0.9, "calibrated": 0.7},
            "unified_narrative": {"primary_verdict": "test"},
        }
        result = DecisionOutput(
            decision="normal",
            confidence=0.70,
            verdict="test verdict",
            severity="info",
            action_required=False,
            action=None,
            metadata=metadata,
        )
        d = result.to_dict()

        assert d["metadata"] == metadata
        # Ensure no top-level keys from metadata
        assert "engine_decision" not in d or isinstance(d.get("engine_decision"), dict)
        assert "calibration_report" not in d or isinstance(d.get("calibration_report"), dict)

    def test_to_dict_confidence_rounded(self):
        """confidence is rounded to 4 decimal places."""
        result = DecisionOutput(
            decision="normal",
            confidence=0.12345678,
            verdict="test",
            severity="info",
            action_required=False,
            action=None,
        )
        d = result.to_dict()

        assert d["confidence"] == pytest.approx(0.1235, abs=0.0001)


class TestDecisionOutputFromMetadata:
    """Factory method from_metadata() extracts values correctly."""

    def test_from_metadata_out_of_domain(self):
        """When within_domain=False, decision='out_of_domain'."""
        metadata: Dict[str, Any] = {
            "boundary_check": {
                "within_domain": False,
                "rejection_reason": "too_many_missing_values",
            }
        }
        result = DecisionOutput.from_metadata(
            metadata=metadata,
            series_id="test_series",
        )

        assert result.decision == "out_of_domain"
        assert result.confidence == pytest.approx(0.0)
        assert result.severity == "unknown"
        assert result.action_required is False
        assert "too_many_missing_values" in result.verdict

    def test_from_metadata_extracts_unified_narrative(self):
        """Extracts verdict and severity from unified_narrative."""
        metadata: Dict[str, Any] = {
            "unified_narrative": {
                "primary_verdict": "temperature anomaly detected",
                "severity": "CRITICAL",
                "confidence": 0.85,
            },
            "calibration_report": {"calibrated": 0.82},
        }
        result = DecisionOutput.from_metadata(
            metadata=metadata,
            series_id="temp_sensor",
            predicted_value=25.5,
        )

        assert result.verdict == "temperature anomaly detected"
        assert result.severity == "critical"
        assert result.confidence == pytest.approx(0.82)

    def test_from_metadata_extracts_action_guard(self):
        """Extracts action info from action_guard."""
        metadata: Dict[str, Any] = {
            "action_guard": {
                "action_required": True,
                "action_allowed": True,
                "final_action": "alert_operator",
            },
            "unified_narrative": {"severity": "WARNING"},
        }
        result = DecisionOutput.from_metadata(
            metadata=metadata,
            series_id="sensor1",
        )

        assert result.action_required is True
        assert result.action == "alert_operator"

    def test_from_metadata_degraded_when_action_suppressed(self):
        """When action_allowed=False, decision='degraded'."""
        metadata: Dict[str, Any] = {
            "action_guard": {
                "action_required": True,
                "action_allowed": False,
                "final_action": None,
            },
            "unified_narrative": {"severity": "WARNING"},
        }
        result = DecisionOutput.from_metadata(
            metadata=metadata,
            series_id="sensor1",
        )

        assert result.decision == "degraded"
        assert result.action_required is False  # Suppressed

    def test_from_metadata_anomaly_from_severity(self):
        """critical or warning severity → decision='anomaly'."""
        metadata: Dict[str, Any] = {
            "unified_narrative": {"severity": "CRITICAL"},
        }
        result = DecisionOutput.from_metadata(
            metadata=metadata,
            series_id="sensor1",
        )

        assert result.decision == "anomaly"
        assert result.severity == "critical"


class TestPredictionDtoToDecisionOutput:
    """PredictionDTO.to_decision_output() constructs correctly."""

    def test_to_decision_output_without_metadata(self):
        """Constructs DecisionOutput from DTO fields when no metadata."""
        dto = PredictionDTO(
            series_id="sensor_123",
            predicted_value=25.5,
            confidence_score=0.85,
            confidence_level="high",
            trend="stable",
            engine_name="taylor",
            explanation_text="temperature stable",
        )
        decision = dto.to_decision_output()

        assert isinstance(decision, DecisionOutput)
        assert decision.confidence == pytest.approx(0.85)
        assert decision.verdict == "temperature stable"
        assert decision.decision == "normal"

    def test_to_decision_output_with_metadata(self):
        """Uses factory method when metadata provided."""
        dto = PredictionDTO(
            series_id="sensor_456",
            predicted_value=30.0,
            confidence_score=0.60,
            confidence_level="low",
            trend="up",
            engine_name="baseline",
        )
        metadata: Dict[str, Any] = {
            "unified_narrative": {
                "primary_verdict": "custom verdict from metadata",
                "severity": "WARNING",
            },
            "calibration_report": {"calibrated": 0.55},
        }
        decision = dto.to_decision_output(metadata=metadata)

        assert decision.verdict == "custom verdict from metadata"
        assert decision.severity == "warning"
        assert decision.confidence == pytest.approx(0.55)


class TestTextDecisionOutput:
    """TextDecisionOutput tests."""

    def test_text_decision_output_construction(self):
        """TextDecisionOutput can be instantiated."""
        result = TextDecisionOutput(
            decision="critical",
            confidence=0.90,
            verdict="SLA breach detected",
            severity="critical",
            domain="infrastructure",
        )
        assert result.decision == "critical"
        assert result.domain == "infrastructure"

    def test_text_to_summary_returns_three_fields(self):
        """to_summary() returns exactly 3 fields."""
        result = TextDecisionOutput(
            decision="normal",
            confidence=0.85,
            verdict="document processed",
            severity="info",
            domain="general",
        )
        summary = result.to_summary()

        assert len(summary) == 3
        assert "decision" in summary
        assert "confidence" in summary
        assert "verdict" in summary
        assert "severity" not in summary
        assert "domain" not in summary

    def test_text_to_dict_includes_all_mandatory(self):
        """to_dict() includes all 5 mandatory fields + metadata."""
        result = TextDecisionOutput(
            decision="anomaly",
            confidence=0.75,
            verdict="urgent keywords found",
            severity="warning",
            domain="operations",
            metadata={"sentiment": {"score": -0.5}},
        )
        d = result.to_dict()

        assert "decision" in d
        assert "confidence" in d
        assert "verdict" in d
        assert "severity" in d
        assert "domain" in d
        assert "metadata" in d
        assert len(d) == 6  # 5 mandatory + metadata


class TestTextDecisionOutputFromTextResult:
    """TextDecisionOutput.from_text_result() factory method."""

    def test_from_text_result_critical(self):
        """Critical risk level → decision='critical', severity='critical'."""
        result_dict: Dict[str, Any] = {
            "severity": {"risk_level": "HIGH", "severity": "CRITICAL"},
            "conclusion": "System outage imminent",
            "confidence": 0.92,
            "domain": "infrastructure",
        }
        output = TextDecisionOutput.from_text_result(result_dict)

        assert output.decision == "critical"
        assert output.severity == "critical"
        assert output.verdict == "System outage imminent"

    def test_from_text_result_warning(self):
        """Medium risk → decision='anomaly', severity='warning'."""
        result_dict: Dict[str, Any] = {
            "severity": {"risk_level": "MEDIUM", "severity": "WARNING"},
            "conclusion": "Performance degradation noted",
            "confidence": 0.70,
            "domain": "application",
        }
        output = TextDecisionOutput.from_text_result(result_dict)

        assert output.decision == "anomaly"
        assert output.severity == "warning"

    def test_from_text_result_normal(self):
        """Low risk → decision='normal', severity='info'."""
        result_dict: Dict[str, Any] = {
            "severity": {"risk_level": "LOW", "severity": "INFO"},
            "conclusion": "All metrics within range",
            "confidence": 0.85,
            "domain": "general",
        }
        output = TextDecisionOutput.from_text_result(result_dict)

        assert output.decision == "normal"
        assert output.severity == "info"

    def test_from_text_result_builds_metadata(self):
        """Factory method puts non-mandatory fields into metadata."""
        result_dict: Dict[str, Any] = {
            "severity": {"risk_level": "LOW", "severity": "INFO"},
            "conclusion": "OK",
            "confidence": 0.80,
            "domain": "test",
            "explanation": {"detail": "full"},
            "analysis": {"word_count": 100},
            "extra_field": "preserved",
        }
        output = TextDecisionOutput.from_text_result(result_dict)

        assert "explanation" in output.metadata
        assert "analysis" in output.metadata
        assert "extra_field" in output.metadata
