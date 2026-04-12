"""Tests for AlertSuppressor."""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.entities.decision import Decision, DecisionContext
from iot_machine_learning.domain.entities.severity import SeverityResult
from iot_machine_learning.domain.services.alert_suppressor import AlertSuppressor


class TestEscalationOverride:
    """Test ESCALATION OVERRIDE rule."""
    
    def test_consecutive_over_5_never_suppressed(self) -> None:
        """Alerts with consecutive_anomalies > 5 are never suppressed."""
        suppressor = AlertSuppressor()
        ctx = DecisionContext(
            series_id="test",
            consecutive_anomalies=6,
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="medium",
                action_required=True,
                recommended_action="Monitor",
            ),
        )
        decision = Decision(
            action="MONITOR",
            priority=3,
            confidence=0.5,
            reason="test",
            strategy_used="test",
        )
        
        result = suppressor.evaluate(ctx, decision)
        
        assert result.should_emit is True
        assert result.reason == "escalation_override"


class TestCriticalityOverride:
    """Test CRITICALITY OVERRIDE rule."""
    
    def test_critical_severity_never_suppressed(self) -> None:
        """Critical severity alerts are never suppressed."""
        suppressor = AlertSuppressor()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="HIGH",
                severity="critical",
                action_required=True,
                recommended_action="Investigate",
            ),
        )
        decision = Decision(
            action="INVESTIGATE",
            priority=2,
            confidence=0.9,
            reason="test",
            strategy_used="test",
        )
        
        result = suppressor.evaluate(ctx, decision)
        
        assert result.should_emit is True
        assert result.reason == "criticality_override"


class TestPriorityEscalation:
    """Test PRIORITY ESCALATION rule."""
    
    def test_higher_priority_not_suppressed(self) -> None:
        """Higher priority (lower number) alerts are not suppressed."""
        # This test would need Redis to track last alert
        # With no Redis, there's no last alert, so it should emit by default
        suppressor = AlertSuppressor()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="medium",
                action_required=True,
                recommended_action="Monitor",
            ),
        )
        decision = Decision(
            action="MONITOR",
            priority=3,
            confidence=0.5,
            reason="test",
            strategy_used="test",
        )
        
        result = suppressor.evaluate(ctx, decision)
        
        # Without Redis and no last alert, should emit by default
        assert result.should_emit is True


class TestDefaultEmit:
    """Test DEFAULT emit behavior."""
    
    def test_default_emit_without_overrides(self) -> None:
        """By default, alerts are emitted."""
        suppressor = AlertSuppressor()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="LOW",
                severity="info",
                action_required=False,
                recommended_action="None",
            ),
        )
        decision = Decision(
            action="LOG_ONLY",
            priority=5,
            confidence=0.1,
            reason="test",
            strategy_used="test",
        )
        
        result = suppressor.evaluate(ctx, decision)
        
        assert result.should_emit is True
        assert result.reason == "default_emit"


class TestSuppressedCount:
    """Test suppressed count tracking."""
    
    def test_suppressed_count_zero_without_redis(self) -> None:
        """Without Redis, suppressed count is always 0."""
        suppressor = AlertSuppressor()
        count = suppressor.get_suppressed_count("test")
        assert count == 0


class TestCanDecide:
    """Test AlertSuppressor doesn't interfere with DecisionEnginePort interface."""
    
    def test_suppressor_is_not_decision_engine(self) -> None:
        """AlertSuppressor is a separate component from DecisionEngine."""
        suppressor = AlertSuppressor()
        
        # Suppressor doesn't implement decide() or can_decide()
        assert not hasattr(suppressor, "decide")
        assert not hasattr(suppressor, "can_decide")


class TestEdgeCases:
    """Test edge cases."""
    
    def test_consecutive_exactly_5_not_override(self) -> None:
        """Exactly 5 consecutive anomalies doesn't trigger escalation override."""
        suppressor = AlertSuppressor()
        ctx = DecisionContext(
            series_id="test",
            consecutive_anomalies=5,
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="medium",
                action_required=True,
                recommended_action="Monitor",
            ),
        )
        decision = Decision(
            action="MONITOR",
            priority=3,
            confidence=0.5,
            reason="test",
            strategy_used="test",
        )
        
        result = suppressor.evaluate(ctx, decision)
        
        # Should be default emit (not escalation override)
        # Note: if it were 6, it would be escalation_override
        assert result.reason == "default_emit"
    
    def test_warning_severity_not_critical(self) -> None:
        """Warning severity doesn't trigger criticality override."""
        suppressor = AlertSuppressor()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="warning",
                action_required=True,
                recommended_action="Monitor",
            ),
        )
        decision = Decision(
            action="MONITOR",
            priority=3,
            confidence=0.5,
            reason="test",
            strategy_used="test",
        )
        
        result = suppressor.evaluate(ctx, decision)
        
        # Should be default emit (not criticality override)
        assert result.reason == "default_emit"
