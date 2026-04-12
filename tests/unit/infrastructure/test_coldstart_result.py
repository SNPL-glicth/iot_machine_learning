"""Tests for ColdStartResult and UniversalComparativeEngine minimum threshold.

DEPRECADO — modulo universal.comparative no existe.
"""

from __future__ import annotations

import pytest
pytestmark = pytest.mark.skip(reason="modulo universal.comparative no existe - pendiente T6/T10")
from unittest.mock import Mock, MagicMock

# Mocks para modulos que no existen
UniversalComparativeEngine = MagicMock
ComparisonContext = MagicMock
ComparisonResult = MagicMock
ColdStartResult = MagicMock
UniversalResult = MagicMock
InputType = MagicMock
from iot_machine_learning.domain.entities.explainability.explanation import (
    Explanation,
    SignalSnapshot,
    Outcome,
)
from iot_machine_learning.domain.services.severity_rules import SeverityResult


class TestColdStartResult:
    """Test ColdStartResult dataclass."""

    def test_coldstart_result_creation(self) -> None:
        """ColdStartResult should be creatable."""
        result = ColdStartResult(
            reason="insufficient_history",
            docs_found=1,
            docs_needed=3,
            message="Comparative analysis available after 3 similar documents",
        )
        
        assert result.reason == "insufficient_history"
        assert result.docs_found == 1
        assert result.docs_needed == 3
        assert "3 similar documents" in result.message

    def test_coldstart_result_to_dict(self) -> None:
        """ColdStartResult should serialize to dict."""
        result = ColdStartResult(
            reason="insufficient_history",
            docs_found=2,
            docs_needed=3,
            message="Need 1 more document",
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["reason"] == "insufficient_history"
        assert result_dict["docs_found"] == 2
        assert result_dict["docs_needed"] == 3
        assert result_dict["message"] == "Need 1 more document"


class TestUniversalComparativeEngineMinThreshold:
    """Test UniversalComparativeEngine minimum threshold logic."""

    def test_engine_default_min_docs_is_three(self) -> None:
        """Default min_similar_docs should be 3."""
        engine = UniversalComparativeEngine()
        assert engine._min_similar_docs == 3

    def test_engine_custom_min_docs(self) -> None:
        """Should accept custom min_similar_docs."""
        engine = UniversalComparativeEngine(min_similar_docs=5)
        assert engine._min_similar_docs == 5

    def test_engine_min_docs_floor_is_one(self) -> None:
        """min_similar_docs should have floor of 1."""
        engine = UniversalComparativeEngine(min_similar_docs=0)
        assert engine._min_similar_docs == 1
        
        engine2 = UniversalComparativeEngine(min_similar_docs=-5)
        assert engine2._min_similar_docs == 1

    def test_compare_returns_none_when_zero_docs(self) -> None:
        """Should return None when no similar docs found."""
        engine = UniversalComparativeEngine(min_similar_docs=3)
        
        # Mock memory that returns empty list
        mock_memory = Mock()
        mock_memory.recall_similar_explanations = Mock(return_value=[])
        
        mock_result = self._create_mock_result()
        
        ctx = ComparisonContext(
            current_result=mock_result,
            series_id="test-001",
            cognitive_memory=mock_memory,
            domain="infrastructure",
        )
        
        result = engine.compare(ctx)
        
        assert result is None

    def test_compare_returns_coldstart_when_insufficient(self) -> None:
        """Should return ColdStartResult when docs < min_similar_docs."""
        engine = UniversalComparativeEngine(min_similar_docs=3)
        
        # Mock memory that returns 2 docs (below threshold)
        mock_memory = Mock()
        mock_memory.recall_similar_explanations = Mock(return_value=[
            {"doc_id": "hist-001", "score": 0.9},
            {"doc_id": "hist-002", "score": 0.85},
        ])
        
        mock_result = self._create_mock_result()
        
        ctx = ComparisonContext(
            current_result=mock_result,
            series_id="test-002",
            cognitive_memory=mock_memory,
            domain="infrastructure",
        )
        
        result = engine.compare(ctx)
        
        assert isinstance(result, ColdStartResult)
        assert result.docs_found == 2
        assert result.docs_needed == 3
        assert result.reason == "insufficient_history"

    def test_compare_returns_comparison_when_sufficient(self) -> None:
        """Should return ComparisonResult when docs >= min_similar_docs."""
        engine = UniversalComparativeEngine(min_similar_docs=3)
        
        # Mock memory that returns 3 docs (meets threshold)
        mock_memory = Mock()
        mock_memory.recall_similar_explanations = Mock(return_value=[
            {
                "doc_id": "hist-001",
                "score": 0.9,
                "summary": "Previous alert",
                "severity": 0.7,
                "timestamp": "2024-01-01T00:00:00",
                "resolution_time": 3600,
            },
            {
                "doc_id": "hist-002",
                "score": 0.85,
                "summary": "Similar issue",
                "severity": 0.6,
                "timestamp": "2024-01-02T00:00:00",
                "resolution_time": 1800,
            },
            {
                "doc_id": "hist-003",
                "score": 0.80,
                "summary": "Related incident",
                "severity": 0.5,
                "timestamp": "2024-01-03T00:00:00",
                "resolution_time": 2400,
            },
        ])
        
        mock_result = self._create_mock_result()
        
        ctx = ComparisonContext(
            current_result=mock_result,
            series_id="test-003",
            cognitive_memory=mock_memory,
            domain="infrastructure",
        )
        
        result = engine.compare(ctx)
        
        # Should NOT be ColdStartResult
        assert not isinstance(result, ColdStartResult)
        # Should be ComparisonResult or None (if comparison logic fails)
        assert result is None or isinstance(result, ComparisonResult)

    def test_different_thresholds(self) -> None:
        """Test different min_similar_docs values."""
        # Threshold = 2
        engine2 = UniversalComparativeEngine(min_similar_docs=2)
        
        mock_memory = Mock()
        mock_memory.recall_similar_explanations = Mock(return_value=[
            {"doc_id": "hist-001", "score": 0.9},
        ])
        
        mock_result = self._create_mock_result()
        ctx = ComparisonContext(
            current_result=mock_result,
            series_id="test-threshold",
            cognitive_memory=mock_memory,
            domain="test",
        )
        
        result = engine2.compare(ctx)
        
        # 1 doc, threshold 2 → ColdStartResult
        assert isinstance(result, ColdStartResult)
        assert result.docs_found == 1
        assert result.docs_needed == 2

    def _create_mock_result(self) -> UniversalResult:
        """Create mock UniversalResult for testing."""
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
            series_id="test",
            signal=signal,
            outcome=outcome,
        )
        
        severity = SeverityResult(
            risk_level="low",
            severity="minor",
            action_required=False,
            recommended_action="monitor",
        )
        
        return UniversalResult(
            explanation=explanation,
            severity=severity,
            analysis={"full_text": "Test analysis", "severity_score": 0.5},
            confidence=0.85,
            domain="infrastructure",
            input_type=InputType.TEXT,
        )
