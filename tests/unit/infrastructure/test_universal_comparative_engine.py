"""Tests for UniversalComparativeEngine."""

from __future__ import annotations

import pytest
from unittest.mock import Mock

from iot_machine_learning.infrastructure.ml.cognitive.universal import (
    UniversalComparativeEngine,
    ComparisonContext,
    UniversalResult,
    UniversalInput,
    InputType,
)
from iot_machine_learning.domain.entities.explainability.explanation import (
    Explanation,
    SignalSnapshot,
    Outcome,
)
from iot_machine_learning.domain.services.severity_rules import SeverityResult


class TestUniversalComparativeEngineBasic:
    """Basic functionality tests."""

    def test_engine_instantiates(self) -> None:
        """Engine should instantiate without errors."""
        engine = UniversalComparativeEngine()
        assert engine is not None

    def test_compare_without_memory_returns_none(self) -> None:
        """Without cognitive memory, should return None gracefully."""
        engine = UniversalComparativeEngine()
        
        # Create mock result
        signal = SignalSnapshot(
            n_points=10,
            mean=100.0,
            std=5.0,
            noise_ratio=0.05,
            slope=0.1,
            curvature=0.01,
            regime="stable",
            dt=1.0,
        )
        outcome = Outcome(
            kind="analysis",
            predicted_value=100.0,
            confidence=0.85,
            trend="stable",
        )
        explanation = Explanation(
            series_id="test-001",
            signal=signal,
            outcome=outcome,
        )
        
        severity = SeverityResult(
            risk_level="low",
            severity="minor",
            action_required=False,
            recommended_action="monitor",
        )
        
        mock_result = UniversalResult(
            explanation=explanation,
            severity=severity,
            analysis={"test": "data"},
            confidence=0.85,
            domain="infrastructure",
            input_type=InputType.TEXT,
        )
        
        ctx = ComparisonContext(
            current_result=mock_result,
            series_id="test-001",
            cognitive_memory=None,  # No memory
            domain="infrastructure",
        )
        
        result = engine.compare(ctx)
        
        assert result is None

    def test_compare_with_empty_memory_returns_none(self) -> None:
        """With memory but no matches, should return None."""
        engine = UniversalComparativeEngine()
        
        # Mock cognitive memory that returns empty list
        mock_memory = Mock()
        mock_memory.recall_similar_explanations = Mock(return_value=[])
        
        signal = SignalSnapshot(
            n_points=10,
            mean=100.0,
            std=5.0,
            noise_ratio=0.05,
            slope=0.1,
            curvature=0.01,
            regime="stable",
            dt=1.0,
        )
        outcome = Outcome(
            kind="analysis",
            predicted_value=100.0,
            confidence=0.85,
            trend="stable",
        )
        explanation = Explanation(
            series_id="test-002",
            signal=signal,
            outcome=outcome,
        )
        
        severity = SeverityResult(
            risk_level="low",
            severity="minor",
            action_required=False,
            recommended_action="monitor",
        )
        
        mock_result = UniversalResult(
            explanation=explanation,
            severity=severity,
            analysis={"full_text": "Test message"},
            confidence=0.85,
            domain="infrastructure",
            input_type=InputType.TEXT,
        )
        
        ctx = ComparisonContext(
            current_result=mock_result,
            series_id="test-002",
            cognitive_memory=mock_memory,
            domain="infrastructure",
        )
        
        result = engine.compare(ctx)
        
        # Should return None when no historical matches found
        assert result is None


class TestUniversalComparativeEngineWithMockMemory:
    """Test with mocked memory responses."""

    def test_compare_with_historical_matches(self) -> None:
        """Compare should work with valid historical data."""
        engine = UniversalComparativeEngine()
        
        # Mock cognitive memory with historical matches
        mock_memory = Mock()
        mock_memory.recall_similar_explanations = Mock(return_value=[
            {
                "doc_id": "hist-001",
                "score": 0.9,
                "summary": "Previous high CPU alert",
                "severity": 0.7,
                "timestamp": "2024-01-01T00:00:00",
                "resolution_time": 3600,
            },
            {
                "doc_id": "hist-002",
                "score": 0.85,
                "summary": "Similar infrastructure issue",
                "severity": 0.6,
                "timestamp": "2024-01-02T00:00:00",
                "resolution_time": 1800,
            },
        ])
        
        signal = SignalSnapshot(
            n_points=10,
            mean=100.0,
            std=5.0,
            noise_ratio=0.05,
            slope=0.1,
            curvature=0.01,
            regime="stable",
            dt=1.0,
        )
        outcome = Outcome(
            kind="analysis",
            predicted_value=100.0,
            confidence=0.85,
            trend="stable",
        )
        explanation = Explanation(
            series_id="test-003",
            signal=signal,
            outcome=outcome,
        )
        
        severity = SeverityResult(
            risk_level="medium",
            severity="moderate",
            action_required=True,
            recommended_action="investigate",
        )
        
        mock_result = UniversalResult(
            explanation=explanation,
            severity=severity,
            analysis={
                "full_text": "High CPU usage detected",
                "severity_score": 0.8,
                "urgency_score": 0.7,
            },
            confidence=0.85,
            domain="infrastructure",
            input_type=InputType.TEXT,
        )
        
        ctx = ComparisonContext(
            current_result=mock_result,
            series_id="test-003",
            cognitive_memory=mock_memory,
            domain="infrastructure",
        )
        
        result = engine.compare(ctx)
        
        # Should return ComparisonResult
        assert result is not None
        assert hasattr(result, "severity_delta_pct")
        assert hasattr(result, "urgency_delta_pct")
        assert hasattr(result, "topic_overlap_pct")
        assert hasattr(result, "delta_conclusion")
        assert len(result.top_similar) >= 0


class TestUniversalComparativeEngineOutputStructure:
    """Test output structure and serialization."""

    def test_comparison_result_to_dict(self) -> None:
        """ComparisonResult should serialize to dict."""
        from iot_machine_learning.infrastructure.ml.cognitive.universal.comparative.types import (
            ComparisonResult,
        )
        
        result = ComparisonResult(
            severity_delta_pct=25.5,
            urgency_delta_pct=-10.2,
            topic_overlap_pct=75.0,
            top_similar=[
                {"doc_id": "test-001", "score": 0.9},
                {"doc_id": "test-002", "score": 0.85},
            ],
            delta_conclusion="Current incident 25% more severe than historical average",
            resolution_probability=0.85,
            estimated_resolution_time="1-2 hours",
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "severity_delta_pct" in result_dict
        assert "urgency_delta_pct" in result_dict
        assert "topic_overlap_pct" in result_dict
        assert "top_similar" in result_dict
        assert "delta_conclusion" in result_dict
        assert "resolution_probability" in result_dict
        assert "estimated_resolution_time" in result_dict


class TestUniversalComparativeEngineGracefulFail:
    """Test graceful failure handling."""

    def test_compare_with_exception_returns_none(self) -> None:
        """Exception during comparison should return None, not crash."""
        engine = UniversalComparativeEngine()
        
        # Mock memory that raises exception
        mock_memory = Mock()
        mock_memory.recall_similar_explanations = Mock(
            side_effect=Exception("Memory service unavailable")
        )
        
        signal = SignalSnapshot(
            n_points=10,
            mean=100.0,
            std=5.0,
            noise_ratio=0.05,
            slope=0.1,
            curvature=0.01,
            regime="stable",
            dt=1.0,
        )
        outcome = Outcome(
            kind="analysis",
            predicted_value=100.0,
            confidence=0.85,
            trend="stable",
        )
        explanation = Explanation(
            series_id="test-fail-001",
            signal=signal,
            outcome=outcome,
        )
        
        severity = SeverityResult(
            risk_level="low",
            severity="minor",
            action_required=False,
            recommended_action="monitor",
        )
        
        mock_result = UniversalResult(
            explanation=explanation,
            severity=severity,
            analysis={"test": "data"},
            confidence=0.85,
            domain="infrastructure",
            input_type=InputType.TEXT,
        )
        
        ctx = ComparisonContext(
            current_result=mock_result,
            series_id="test-fail-001",
            cognitive_memory=mock_memory,
            domain="infrastructure",
        )
        
        # Should not raise exception, should return None
        result = engine.compare(ctx)
        assert result is None
