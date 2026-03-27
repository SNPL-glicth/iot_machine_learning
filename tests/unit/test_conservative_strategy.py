"""Tests for ConservativeStrategy.

Tests the conservative decision hierarchy:
1. severity=critical OR pattern contains escalation → escalate
2. confidence > 0.8 → intervene
3. default → investigate
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.entities.decision import DecisionContext
from iot_machine_learning.domain.entities.decision.priority import Priority
from iot_machine_learning.domain.entities.results.anomaly import AnomalySeverity
from iot_machine_learning.domain.services.severity_rules import SeverityResult
from iot_machine_learning.infrastructure.ml.cognitive.decision import ConservativeStrategy


class TestConservativeStrategyBasic:
    """Test basic ConservativeStrategy functionality."""

    def test_strategy_name(self):
        """Strategy reports correct name."""
        strategy = ConservativeStrategy()
        assert strategy.strategy_name == "conservative"

    def test_version(self):
        """Strategy reports version."""
        strategy = ConservativeStrategy(version="2.0.0")
        assert strategy.version == "2.0.0"

    def test_default_version(self):
        """Default version is 1.0.0."""
        strategy = ConservativeStrategy()
        assert strategy.version == "1.0.0"

    def test_can_decide_with_severity(self):
        """Can decide when severity present."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="LOW",
                severity="info",
                action_required=False,
                recommended_action="OK",
            ),
        )
        assert strategy.can_decide(ctx) is True

    def test_can_decide_with_patterns(self):
        """Can decide when patterns present (no severity)."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            patterns=[{"pattern_type": "drift", "severity_hint": "info"}],
        )
        assert strategy.can_decide(ctx) is True

    def test_can_decide_empty_context(self):
        """Cannot decide with completely empty context."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(series_id="test")
        assert strategy.can_decide(ctx) is False


class TestConservativeDecisionHierarchy:
    """Test conservative decision hierarchy rules."""

    def test_critical_severity_escalates(self):
        """Rule 1: Critical severity → escalate."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="HIGH",
                severity="critical",
                action_required=True,
                recommended_action="Critical condition!",
            ),
            confidence=0.5,  # Low confidence shouldn't matter
        )
        
        decision = strategy.decide(ctx)
        
        assert decision.action == "escalate"
        assert decision.priority == Priority.CRITICAL
        assert decision.is_actionable is True
        assert "critical" in decision.reason.lower()

    def test_critical_pattern_escalates(self):
        """Rule 1: Pattern with severity_hint=critical → escalate."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            patterns=[
                {"pattern_type": "spike", "severity_hint": "critical", "confidence": 0.9},
            ],
        )
        
        decision = strategy.decide(ctx)
        
        assert decision.action == "escalate"
        assert decision.priority == Priority.CRITICAL

    def test_high_confidence_intervenes(self):
        """Rule 2: confidence > 0.8 → intervene."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            confidence=0.85,
            is_anomaly=True,
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="warning",
                action_required=True,
                recommended_action="Check this",
            ),
        )
        
        decision = strategy.decide(ctx)
        
        assert decision.action == "intervene"
        assert decision.priority == Priority.HIGH
        assert "0.85" in decision.reason or "0.8" in decision.reason

    def test_exact_threshold_intervenes(self):
        """Boundary: confidence = 0.8 should NOT intervene (must be >)."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            confidence=0.8,  # Exactly at threshold
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="warning",
                action_required=False,
                recommended_action="OK",
            ),
        )
        
        decision = strategy.decide(ctx)
        
        # Should default to investigate since not > 0.8
        assert decision.action == "investigate"

    def test_default_investigate(self):
        """Rule 3: Default when no critical/high-confidence conditions."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            confidence=0.6,
            severity=SeverityResult(
                risk_level="LOW",
                severity="info",
                action_required=False,
                recommended_action="All normal",
            ),
        )
        
        decision = strategy.decide(ctx)
        
        assert decision.action == "investigate"
        assert decision.priority == Priority.MEDIUM
        assert "conservative" in decision.reason.lower() or "false positive" in decision.reason.lower()

    def test_empty_context_safe_fallback(self):
        """Empty context returns investigate (safe default)."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(series_id="test")
        
        # Even though can_decide returns False, decide() should still work
        decision = strategy.decide(ctx)
        
        # Should be safe default
        assert decision.action in ["investigate", "monitor"]


class TestConservativeSimulatedOutcomes:
    """Test simulated outcomes generation."""

    def test_generates_three_scenarios(self):
        """Conservative strategy generates 3 scenarios by default."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            confidence=0.5,
        )
        
        decision = strategy.decide(ctx)
        
        assert len(decision.simulated_outcomes) == 3
        scenario_names = [o.scenario_name for o in decision.simulated_outcomes]
        assert "do_nothing" in scenario_names
        assert "act_conservative" in scenario_names
        assert "worst_case" in scenario_names

    def test_uses_existing_monte_carlo(self):
        """Uses context monte_carlo_outcomes if available."""
        from iot_machine_learning.domain.entities.decision import SimulatedOutcome
        
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            confidence=0.5,
            monte_carlo_outcomes=[
                SimulatedOutcome(scenario_name="mc_1", probability=0.3, expected_risk=0.7),
                SimulatedOutcome(scenario_name="mc_2", probability=0.7, expected_risk=0.4),
            ],
        )
        
        decision = strategy.decide(ctx)
        
        # Should use provided outcomes, not generate new ones
        assert len(decision.simulated_outcomes) == 2
        assert decision.simulated_outcomes[0].scenario_name == "mc_1"

    def test_worst_case_has_safety_margin(self):
        """Worst case risk has safety margin applied."""
        strategy = ConservativeStrategy(safety_margin=1.5)
        ctx = DecisionContext(
            series_id="test",
            confidence=0.5,
            is_anomaly=True,
            anomaly_score=0.5,  # Base risk factor
        )
        
        decision = strategy.decide(ctx)
        
        worst = [o for o in decision.simulated_outcomes if o.scenario_name == "worst_case"][0]
        # 0.5 base * 1.5 margin = 0.75, capped at 1.0
        assert worst.expected_risk == 0.75

    def test_worst_case_risk_in_reason(self):
        """Worst case risk is included in decision reason."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="HIGH",
                severity="critical",
                action_required=True,
                recommended_action="Critical!",
            ),
        )
        
        decision = strategy.decide(ctx)
        
        assert "risk:" in decision.reason.lower() or "worst-case" in decision.reason.lower()


class TestConservativeConfidenceCalculation:
    """Test confidence calculation logic."""

    def test_confidence_has_floor(self):
        """Conservative confidence never below 0.6."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            confidence=0.1,  # Very low input confidence
        )
        
        decision = strategy.decide(ctx)
        
        assert decision.confidence >= 0.6

    def test_confidence_has_ceiling(self):
        """Conservative confidence never above 0.95."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            confidence=1.0,  # Perfect input confidence
        )
        
        decision = strategy.decide(ctx)
        
        assert decision.confidence <= 0.95

    def test_sparse_evidence_reduces_confidence(self):
        """Sparse evidence reduces confidence (but still >= 0.6)."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            confidence=0.9,
            # No patterns, no severity = sparse evidence
        )
        
        decision = strategy.decide(ctx)
        
        # Should be reduced from 0.9 due to sparse evidence
        assert decision.confidence < 0.9
        assert decision.confidence >= 0.6

    def test_rich_evidence_maintains_confidence(self):
        """Rich evidence maintains high confidence."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            confidence=0.9,
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="warning",
                action_required=True,
                recommended_action="Check",
            ),
            patterns=[
                {"pattern_type": "spike", "severity_hint": "warning"},
                {"pattern_type": "drift", "severity_hint": "info"},
            ],
        )
        
        decision = strategy.decide(ctx)
        
        # Rich evidence should maintain confidence near input
        assert decision.confidence >= 0.8


class TestConservativeSourceOutputs:
    """Test source ML outputs reference."""

    def test_source_outputs_include_config(self):
        """Source outputs include strategy config."""
        strategy = ConservativeStrategy(
            confidence_threshold=0.75,
            safety_margin=1.3,
        )
        ctx = DecisionContext(series_id="sensor-42")
        
        decision = strategy.decide(ctx)
        
        assert "strategy_config" in decision.source_ml_outputs
        config = decision.source_ml_outputs["strategy_config"]
        assert config["confidence_threshold"] == 0.75
        assert config["safety_margin"] == 1.3

    def test_source_outputs_include_context_data(self):
        """Source outputs include relevant context fields."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="sensor-42",
            confidence=0.85,
            is_anomaly=True,
            anomaly_score=0.7,
            trend="increasing",
            domain="infrastructure",
            patterns=[{"type": "spike"}],
        )
        
        decision = strategy.decide(ctx)
        
        assert decision.source_ml_outputs["series_id"] == "sensor-42"
        assert decision.source_ml_outputs["confidence"] == 0.85
        assert decision.source_ml_outputs["is_anomaly"] is True
        assert decision.source_ml_outputs["anomaly_score"] == 0.7
        assert decision.source_ml_outputs["trend"] == "increasing"
        assert decision.source_ml_outputs["domain"] == "infrastructure"
        assert decision.source_ml_outputs["pattern_count"] == 1


class TestConservativeDecideSafe:
    """Test fail-safe wrapper."""

    def test_decide_safe_returns_decision_on_success(self):
        """decide_safe returns valid decision normally."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            confidence=0.9,  # High confidence
        )
        
        decision = strategy.decide_safe(ctx)
        
        assert isinstance(decision, type(strategy.decide(ctx)))
        assert decision.action is not None

    def test_decide_safe_returns_noop_on_exception(self):
        """decide_safe returns noop on exception."""
        from unittest.mock import patch
        
        strategy = ConservativeStrategy()
        ctx = DecisionContext(series_id="test")
        
        # Force an exception in decide()
        with patch.object(strategy, '_is_critical_condition', side_effect=Exception("boom")):
            decision = strategy.decide_safe(ctx, fallback_reason="Test failure")
        
        assert decision.action == "monitor"
        assert decision.priority == Priority.LOW
        assert "Test failure" in decision.reason


class TestConservativeRiskEstimation:
    """Test risk estimation logic."""

    def test_risk_estimation_considers_anomaly(self):
        """Risk estimation includes anomaly score."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            is_anomaly=True,
            anomaly_score=0.9,
        )
        
        decision = strategy.decide(ctx)
        
        # Should have high risk in scenarios due to anomaly
        worst = [o for o in decision.simulated_outcomes if o.scenario_name == "worst_case"][0]
        assert worst.expected_risk > 0.8  # 0.9 * 1.2 safety margin, capped

    def test_risk_estimation_considers_severity(self):
        """Risk estimation includes severity level."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="HIGH",
                severity="warning",
                action_required=True,
                recommended_action="Check",
            ),
        )
        
        decision = strategy.decide(ctx)
        
        # Warning severity = 0.6 base risk
        base = [o for o in decision.simulated_outcomes if o.scenario_name == "do_nothing"][0]
        assert base.expected_risk >= 0.5

    def test_risk_estimation_considers_patterns(self):
        """Risk estimation includes pattern hints."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(
            series_id="test",
            patterns=[
                {"pattern_type": "spike", "severity_hint": "critical"},
            ],
        )
        
        decision = strategy.decide(ctx)
        
        # Critical pattern hint = 0.9 risk
        base = [o for o in decision.simulated_outcomes if o.scenario_name == "do_nothing"][0]
        assert base.expected_risk >= 0.8

    def test_risk_estimation_default_when_no_data(self):
        """Default moderate risk when no data available."""
        strategy = ConservativeStrategy()
        ctx = DecisionContext(series_id="test")
        
        decision = strategy.decide(ctx)
        
        base = [o for o in decision.simulated_outcomes if o.scenario_name == "do_nothing"][0]
        assert base.expected_risk == 0.3  # Default risk
