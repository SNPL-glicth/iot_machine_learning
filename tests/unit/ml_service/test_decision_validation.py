"""Tests for DecisionEngine validation against analysis."""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any

from iot_machine_learning.ml_service.api.services.analysis.decision_engine_service import DecisionEngineService


@dataclass
class MockSeverity:
    """Mock severity object."""
    severity: str


@dataclass
class MockAnalysisResult:
    """Mock analysis result."""
    severity: Any


class TestDecisionValidation:
    """Test suite for decision validation."""
    
    def test_prevent_downgrade_of_critical_alert(self):
        """Test that DecisionEngine cannot lower a critical alert."""
        service = DecisionEngineService()
        
        decision = {"action": "ignore", "priority": "low"}
        analysis = MockAnalysisResult(severity=MockSeverity("critical"))
        
        validated = service.validate_against_analysis(decision, analysis)
        
        # Should override to escalate
        assert validated["action"] == "escalate"
        assert "decision_override_reason" in validated
        assert "critical" in validated["decision_override_reason"]
    
    def test_prevent_upgrade_when_analysis_is_low(self):
        """Test that DecisionEngine cannot escalate when analysis is low."""
        service = DecisionEngineService()
        
        decision = {"action": "escalate", "priority": "high"}
        analysis = MockAnalysisResult(severity=MockSeverity("low"))
        
        validated = service.validate_against_analysis(decision, analysis)
        
        # Should override to monitor
        assert validated["action"] == "monitor"
        assert "decision_override_reason" in validated
        assert "low" in validated["decision_override_reason"]
    
    def test_prevent_stable_when_severity_is_critical(self):
        """Test that DecisionEngine cannot suggest stable when severity is critical."""
        service = DecisionEngineService()
        
        decision = {"action": "stable", "priority": "low"}
        analysis = MockAnalysisResult(severity=MockSeverity("critical"))
        
        validated = service.validate_against_analysis(decision, analysis)
        
        # Should override to investigate
        assert validated["action"] == "investigate"
        assert "decision_override_reason" in validated
        assert "stable" in validated["decision_override_reason"]
    
    def test_allow_aligned_decision(self):
        """Test that aligned decisions pass through unchanged."""
        service = DecisionEngineService()
        
        decision = {"action": "escalate", "priority": "high"}
        analysis = MockAnalysisResult(severity=MockSeverity("critical"))
        
        validated = service.validate_against_analysis(decision, analysis)
        
        # Should pass through unchanged
        assert validated["action"] == "escalate"
        assert "decision_override_reason" not in validated
    
    def test_handle_string_severity(self):
        """Test handling of string severity (not object)."""
        service = DecisionEngineService()
        
        decision = {"action": "ignore", "priority": "low"}
        analysis = MockAnalysisResult(severity="high")  # String, not object
        
        validated = service.validate_against_analysis(decision, analysis)
        
        # Should still override
        assert validated["action"] == "escalate"
        assert "decision_override_reason" in validated
    
    def test_conservative_approach(self):
        """Test that all overrides are conservative (safer)."""
        service = DecisionEngineService()
        
        # Test 1: Critical + ignore → escalate (safer)
        decision1 = {"action": "ignore"}
        analysis1 = MockAnalysisResult(severity=MockSeverity("critical"))
        validated1 = service.validate_against_analysis(decision1, analysis1)
        assert validated1["action"] == "escalate"
        
        # Test 2: Low + escalate → monitor (safer)
        decision2 = {"action": "escalate"}
        analysis2 = MockAnalysisResult(severity=MockSeverity("low"))
        validated2 = service.validate_against_analysis(decision2, analysis2)
        assert validated2["action"] == "monitor"
        
        # Test 3: Warning + stable → investigate (safer)
        decision3 = {"action": "stable"}
        analysis3 = MockAnalysisResult(severity=MockSeverity("warning"))
        validated3 = service.validate_against_analysis(decision3, analysis3)
        assert validated3["action"] == "investigate"
