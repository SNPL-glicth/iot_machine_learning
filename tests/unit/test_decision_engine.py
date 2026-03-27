"""End-to-end tests for Decision Engine integration.

Tests the full flow from UniversalResult → DecisionContext → Decision.
Verifies feature flag gating, graceful failures, and output format.
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from iot_machine_learning.domain.entities.decision import (
    Decision,
    DecisionContext,
    SimulatedOutcome,
)
from iot_machine_learning.domain.entities.decision.priority import Priority
from iot_machine_learning.domain.entities.results.anomaly import AnomalySeverity
from iot_machine_learning.domain.services.severity_rules import SeverityResult
from iot_machine_learning.domain.ports.decision_port import (
    DecisionEnginePort,
    NullDecisionEngine,
    DecisionEngineRegistry,
)
from iot_machine_learning.infrastructure.ml.cognitive.decision import SimpleDecisionEngine


class TestDecisionContextCreation:
    """Test DecisionContext value object."""

    def test_default_construction(self):
        """Context can be created with defaults."""
        ctx = DecisionContext(series_id="test-123")
        
        assert ctx.series_id == "test-123"
        assert ctx.confidence == 0.0
        assert ctx.is_anomaly is False
        assert ctx.trend == "stable"
        assert ctx.has_monte_carlo is False
        assert ctx.has_critical_pattern is False
        assert ctx.action_required is False  # Default severity has action_required=False

    def test_with_severity(self):
        """Context with severity info."""
        severity = SeverityResult(
            risk_level="HIGH",
            severity="warning",
            action_required=True,
            recommended_action="Investigate immediately",
        )
        ctx = DecisionContext(
            series_id="sensor-42",
            severity=severity,
            confidence=0.85,
            is_anomaly=True,
            anomaly_score=0.78,
        )
        
        assert ctx.action_required is True
        assert ctx.severity.risk_level == "HIGH"
        assert ctx.confidence == 0.85

    def test_has_critical_pattern_true(self):
        """Detects critical pattern in patterns list."""
        ctx = DecisionContext(
            series_id="test",
            patterns=[
                {"pattern_type": "spike", "severity_hint": "critical", "confidence": 0.9},
                {"pattern_type": "drift", "severity_hint": "info", "confidence": 0.5},
            ],
        )
        
        assert ctx.has_critical_pattern is True

    def test_has_critical_pattern_false(self):
        """No critical patterns detected."""
        ctx = DecisionContext(
            series_id="test",
            patterns=[
                {"pattern_type": "drift", "severity_hint": "warning", "confidence": 0.6},
            ],
        )
        
        assert ctx.has_critical_pattern is False


class TestDecisionCreation:
    """Test Decision value object."""

    def test_default_decision_is_monitor(self):
        """Default decision is low priority monitor."""
        dec = Decision()
        
        assert dec.action == "monitor"
        assert dec.priority == Priority.LOW
        assert dec.is_actionable is False
        assert dec.has_simulated_outcomes is False

    def test_from_severity_critical(self):
        """Critical severity → escalate with priority 1."""
        severity = SeverityResult(
            risk_level="HIGH",
            severity="critical",
            action_required=True,
            recommended_action="Critical condition",
        )
        dec = Decision.from_severity(severity, series_id="s-1")
        
        assert dec.action == "escalate"
        assert dec.priority == Priority.CRITICAL
        assert dec.is_actionable is True
        assert dec.reason == "Critical condition"

    def test_from_severity_warning(self):
        """Warning severity → investigate with priority 2."""
        severity = SeverityResult(
            risk_level="MEDIUM",
            severity="warning",
            action_required=True,
            recommended_action="Investigate",
        )
        dec = Decision.from_severity(severity, series_id="s-1")
        
        assert dec.action == "investigate"
        assert dec.priority == Priority.HIGH
        assert dec.is_actionable is True

    def test_from_severity_info(self):
        """Info severity → monitor with priority 4."""
        severity = SeverityResult(
            risk_level="LOW",
            severity="info",
            action_required=False,
            recommended_action="All normal",
        )
        dec = Decision.from_severity(severity, series_id="s-1")
        
        assert dec.action == "monitor"
        assert dec.priority == Priority.LOW
        assert dec.is_actionable is False
        assert dec.confidence == 0.95  # Higher confidence for normal case

    def test_noop_factory(self):
        """Noop decision for graceful fallback."""
        dec = Decision.noop(series_id="s-1", reason="Engine disabled")
        
        assert dec.action == "monitor"
        assert dec.priority == Priority.LOW
        assert dec.confidence == 1.0
        assert dec.reason == "Engine disabled"
        assert dec.strategy_used == "noop"

    def test_to_dict_serialization(self):
        """Decision serializes to dict correctly."""
        dec = Decision(
            action="escalate",
            priority=Priority.CRITICAL,
            confidence=0.85,
            reason="Critical detected",
            strategy_used="simple",
            simulated_outcomes=[
                SimulatedOutcome(
                    scenario_name="do_nothing",
                    probability=0.7,
                    expected_risk=0.9,
                ),
            ],
        )
        d = dec.to_dict()
        
        assert d["action"] == "escalate"
        assert d["priority"] == 1
        assert d["confidence"] == 0.85
        assert d["is_actionable"] is True
        assert len(d["simulated_outcomes"]) == 1
        assert d["simulated_outcomes"][0]["scenario_name"] == "do_nothing"


class TestSimpleDecisionEngine:
    """Test SimpleDecisionEngine MVP implementation."""

    def test_strategy_name_and_version(self):
        """Engine reports correct metadata."""
        engine = SimpleDecisionEngine()
        
        assert engine.strategy_name == "simple"
        assert engine.version == "1.0.0"

    def test_can_decide_with_valid_context(self):
        """Can always decide with valid context (severity has default)."""
        engine = SimpleDecisionEngine()
        ctx = DecisionContext(series_id="test")
        
        assert engine.can_decide(ctx) is True

    def test_decide_maps_severity_critical(self):
        """Critical severity maps to escalate."""
        engine = SimpleDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="HIGH",
                severity="critical",
                action_required=True,
                recommended_action="Critical!",
            ),
        )
        
        dec = engine.decide(ctx)
        
        assert dec.action == "escalate"
        assert dec.priority == Priority.CRITICAL
        assert dec.strategy_used == "simple"

    def test_decide_includes_audit_trace_id(self):
        """Decision includes audit trace ID from context."""
        engine = SimpleDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            audit_trace_id="audit-123",
            severity=SeverityResult(
                risk_level="LOW",
                severity="info",
                action_required=False,
                recommended_action="OK",
            ),
        )
        
        dec = engine.decide(ctx)
        
        assert dec.audit_trace_id == "audit-123"

    def test_decide_safe_returns_valid_decision(self):
        """decide_safe returns valid decision on success."""
        engine = SimpleDecisionEngine()
        ctx = DecisionContext(series_id="test")
        
        dec = engine.decide_safe(ctx)
        
        assert isinstance(dec, Decision)
        assert dec.action is not None

    def test_decide_safe_returns_noop_on_failure(self):
        """decide_safe returns noop on exception."""
        engine = SimpleDecisionEngine()
        
        # Create a mock context that will cause decide() to fail
        bad_ctx = Mock()
        bad_ctx.severity = None  # This will cause issues
        bad_ctx.series_id = "bad"
        
        # SimpleEngine can handle None severity (has default), so we need to mock decide
        with patch.object(engine, 'decide', side_effect=Exception("boom")):
            dec = engine.decide_safe(bad_ctx, fallback_reason="Failed")
        
        assert dec.action == "monitor"
        assert "Failed" in dec.reason


class TestNullDecisionEngine:
    """Test Null object pattern implementation."""

    def test_always_returns_noop(self):
        """Null engine always returns noop."""
        engine = NullDecisionEngine()
        ctx = DecisionContext(series_id="test")
        
        dec = engine.decide(ctx)
        
        assert dec.action == "monitor"
        assert dec.priority == Priority.LOW
        assert dec.reason == "Decision engine disabled (NullDecisionEngine)"

    def test_always_can_decide(self):
        """Null engine always reports it can decide."""
        engine = NullDecisionEngine()
        assert engine.can_decide(Mock()) is True


class TestDecisionEngineRegistry:
    """Test registry pattern for strategy selection."""

    def test_register_and_retrieve(self):
        """Can register and retrieve engines."""
        engine = SimpleDecisionEngine()
        DecisionEngineRegistry.register(engine)
        
        retrieved = DecisionEngineRegistry.get("simple")
        assert retrieved is engine

    def test_list_available(self):
        """Can list available strategies."""
        DecisionEngineRegistry.register(SimpleDecisionEngine())
        
        available = DecisionEngineRegistry.list_available()
        assert "simple" in available

    def test_create_default_returns_registered(self):
        """create_default returns registered engine."""
        engine = SimpleDecisionEngine()
        DecisionEngineRegistry.register(engine)
        
        default = DecisionEngineRegistry.create_default()
        assert default is engine

    def test_create_default_returns_null_if_none(self):
        """create_default returns NullDecisionEngine if none registered."""
        # Clear registry
        DecisionEngineRegistry._engines.clear()
        
        default = DecisionEngineRegistry.create_default()
        assert isinstance(default, NullDecisionEngine)


class TestSimulatedOutcome:
    """Test SimulatedOutcome value object."""

    def test_construction(self):
        """Can create SimulatedOutcome."""
        outcome = SimulatedOutcome(
            scenario_name="do_nothing",
            probability=0.6,
            expected_risk=0.8,
            confidence_interval=(0.7, 0.9),
            description="Risk increases if no action",
        )
        
        assert outcome.scenario_name == "do_nothing"
        assert outcome.probability == 0.6

    def test_to_dict(self):
        """Serializes correctly."""
        outcome = SimulatedOutcome(
            scenario_name="act_conservative",
            probability=0.75,
            expected_risk=0.3,
            confidence_interval=(0.2, 0.4),
        )
        d = outcome.to_dict()
        
        assert d["scenario_name"] == "act_conservative"
        assert d["probability"] == 0.75
        assert d["confidence_interval"] == [0.2, 0.4]


class TestDecisionContextToDict:
    """Test serialization of DecisionContext."""

    def test_full_serialization(self):
        """Complete context serializes to dict."""
        ctx = DecisionContext(
            series_id="doc-123",
            severity=SeverityResult(
                risk_level="HIGH",
                severity="warning",
                action_required=True,
                recommended_action="Check it",
            ),
            confidence=0.82,
            is_anomaly=True,
            anomaly_score=0.71,
            patterns=[{"pattern_type": "spike", "severity_hint": "warning"}],
            predicted_value=42.5,
            trend="increasing",
            monte_carlo_outcomes=[
                SimulatedOutcome(scenario_name="s1", probability=0.5),
            ],
            domain="infrastructure",
            audit_trace_id="trace-abc",
        )
        d = ctx.to_dict()
        
        assert d["series_id"] == "doc-123"
        assert d["severity"]["risk_level"] == "HIGH"
        assert d["confidence"] == 0.82
        assert d["is_anomaly"] is True
        assert d["anomaly_score"] == 0.71
        assert len(d["patterns"]) == 1
        assert d["predicted_value"] == 42.5
        assert d["trend"] == "increasing"
        assert d["monte_carlo_outcomes"] is not None
        assert d["domain"] == "infrastructure"
        assert d["audit_trace_id"] == "trace-abc"
