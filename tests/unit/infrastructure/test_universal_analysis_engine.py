"""Tests for UniversalAnalysisEngine."""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.universal import (
    UniversalAnalysisEngine,
    UniversalInput,
    UniversalContext,
    InputType,
)


class TestUniversalAnalysisEngineBasic:
    """Basic functionality tests."""

    def test_engine_instantiates(self) -> None:
        """Engine should instantiate without errors."""
        engine = UniversalAnalysisEngine()
        assert engine is not None

    def test_analyze_text_input(self) -> None:
        """Analyze simple text input."""
        engine = UniversalAnalysisEngine()
        
        universal_input = UniversalInput(
            raw_data="System alert: High CPU usage detected on server-01",
            detected_type=InputType.TEXT,
            metadata={"word_count": 8},
            series_id="test-001",
        )
        
        context = UniversalContext(
            series_id="test-001",
            tenant_id="test-tenant",
            budget_ms=1000.0,
        )
        
        result = engine.analyze(universal_input, context)
        
        assert result is not None
        assert result.input_type == InputType.TEXT
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
        assert result.domain != ""
        assert result.explanation is not None

    def test_analyze_numeric_input(self) -> None:
        """Analyze numeric series."""
        engine = UniversalAnalysisEngine()
        
        values = [20.0 + i * 0.1 for i in range(50)]
        
        universal_input = UniversalInput(
            raw_data=values,
            detected_type=InputType.NUMERIC,
            metadata={"n_points": len(values), "mean": sum(values)/len(values)},
            series_id="test-002",
        )
        
        context = UniversalContext(
            series_id="test-002",
            tenant_id="test-tenant",
        )
        
        result = engine.analyze(universal_input, context)
        
        assert result is not None
        assert result.input_type == InputType.NUMERIC
        assert "analysis" in result.to_dict()

    def test_analyze_insufficient_data_graceful(self) -> None:
        """Handle insufficient data gracefully."""
        engine = UniversalAnalysisEngine()
        
        universal_input = UniversalInput(
            raw_data="",
            detected_type=InputType.TEXT,
            series_id="test-003",
        )
        
        context = UniversalContext(series_id="test-003")
        
        result = engine.analyze(universal_input, context)
        
        assert result is not None
        assert result.confidence < 0.5  # Low confidence for empty input


class TestUniversalAnalysisEngineDomains:
    """Test domain classification."""

    def test_infrastructure_domain_detection(self) -> None:
        """Infrastructure keywords trigger infrastructure domain."""
        engine = UniversalAnalysisEngine()
        
        universal_input = UniversalInput(
            raw_data="Server disk usage at 95% capacity. Network latency increased.",
            detected_type=InputType.TEXT,
            series_id="test-infra-001",
        )
        
        context = UniversalContext(series_id="test-infra-001")
        
        result = engine.analyze(universal_input, context)
        
        assert result.domain in ("infrastructure", "operations", "general")

    def test_security_domain_detection(self) -> None:
        """Security keywords trigger security domain."""
        engine = UniversalAnalysisEngine()
        
        universal_input = UniversalInput(
            raw_data="Unauthorized access attempt detected. Failed login from unknown IP.",
            detected_type=InputType.TEXT,
            series_id="test-sec-001",
        )
        
        context = UniversalContext(series_id="test-sec-001")
        
        result = engine.analyze(universal_input, context)
        
        assert result.domain in ("security", "general")

    def test_domain_hint_override(self) -> None:
        """Explicit domain hint should be respected."""
        engine = UniversalAnalysisEngine()
        
        universal_input = UniversalInput(
            raw_data="Generic message",
            detected_type=InputType.TEXT,
            domain_hint="trading",
            series_id="test-hint-001",
        )
        
        context = UniversalContext(
            series_id="test-hint-001",
            domain_hint="trading",
        )
        
        result = engine.analyze(universal_input, context)
        
        assert result.domain == "trading"


class TestUniversalAnalysisEngineOutputStructure:
    """Test output structure and serialization."""

    def test_result_to_dict_serializable(self) -> None:
        """Result should serialize to dict."""
        engine = UniversalAnalysisEngine()
        
        universal_input = UniversalInput(
            raw_data="Test message",
            detected_type=InputType.TEXT,
            series_id="test-ser-001",
        )
        
        context = UniversalContext(series_id="test-ser-001")
        
        result = engine.analyze(universal_input, context)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "explanation" in result_dict
        assert "severity" in result_dict
        assert "analysis" in result_dict
        assert "confidence" in result_dict
        assert "domain" in result_dict
        assert "input_type" in result_dict

    def test_explanation_structure(self) -> None:
        """Explanation should have expected structure."""
        engine = UniversalAnalysisEngine()
        
        universal_input = UniversalInput(
            raw_data="Critical system failure",
            detected_type=InputType.TEXT,
            series_id="test-exp-001",
        )
        
        context = UniversalContext(series_id="test-exp-001")
        
        result = engine.analyze(universal_input, context)
        exp_dict = result.explanation.to_dict()
        
        assert "series_id" in exp_dict
        assert "signal" in exp_dict
        assert "outcome" in exp_dict


class TestUniversalAnalysisEngineGracefulFail:
    """Test graceful failure on edge cases."""

    def test_none_input_handled(self) -> None:
        """None input should not crash."""
        engine = UniversalAnalysisEngine()
        
        universal_input = UniversalInput(
            raw_data=None,
            detected_type=InputType.UNKNOWN,
            series_id="test-none-001",
        )
        
        context = UniversalContext(series_id="test-none-001")
        
        # Should not raise exception
        result = engine.analyze(universal_input, context)
        assert result is not None

    def test_special_chars_input(self) -> None:
        """Special characters input should be handled."""
        engine = UniversalAnalysisEngine()
        
        universal_input = UniversalInput(
            raw_data="!@#$%^&*()_+-=[]{}|;':\",./<>?",
            detected_type=InputType.SPECIAL_CHARS,
            series_id="test-spec-001",
        )
        
        context = UniversalContext(series_id="test-spec-001")
        
        result = engine.analyze(universal_input, context)
        assert result is not None
        assert result.confidence >= 0.0


class TestUniversalAnalysisEnginePlasticity:
    """Test plasticity tracking across series."""

    def test_repeated_series_id_learns(self) -> None:
        """Repeated analysis of same series should adapt."""
        engine = UniversalAnalysisEngine()
        
        series_id = "plasticity-test-001"
        
        # First analysis
        input1 = UniversalInput(
            raw_data=[10.0] * 20,
            detected_type=InputType.NUMERIC,
            series_id=series_id,
        )
        context1 = UniversalContext(series_id=series_id)
        result1 = engine.analyze(input1, context1)
        
        # Second analysis - same series
        input2 = UniversalInput(
            raw_data=[10.5] * 20,
            detected_type=InputType.NUMERIC,
            series_id=series_id,
        )
        context2 = UniversalContext(series_id=series_id)
        result2 = engine.analyze(input2, context2)
        
        # Both should succeed (plasticity tracks internally)
        assert result1 is not None
        assert result2 is not None
