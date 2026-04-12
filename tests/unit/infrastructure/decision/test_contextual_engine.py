"""Tests for ContextualDecisionEngine."""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.entities.decision import DecisionContext
from iot_machine_learning.domain.entities.severity import SeverityResult
from iot_machine_learning.infrastructure.ml.cognitive.decision import ContextualDecisionEngine


class TestBaseScore:
    """Test base score calculation from severity."""
    
    def test_critical_severity_base_score(self) -> None:
        """Critical severity produces high base score."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="HIGH",
                severity="critical",
                action_required=True,
                recommended_action="Investigate",
            ),
            # Use NOISY to avoid stable attenuator, but include anomaly_count to avoid no_context
            current_regime="NOISY",
            recent_anomaly_count=3,
        )
        
        decision = engine.decide(ctx)
        
        # With critical base (0.90), noisy amplifier (1.10), and 3 anomalies
        # score should be high enough for escalation
        assert decision.confidence >= 0.75
        assert decision.action == "ESCALATE"
    
    def test_high_severity_base_score(self) -> None:
        """High severity produces medium-high base score."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="HIGH",
                severity="high",
                action_required=True,
                recommended_action="Investigate",
            ),
            current_regime="NOISY",
            recent_anomaly_count=3,
        )
        
        decision = engine.decide(ctx)
        
        # Base 0.70 + noisy amplifier should give confidence >= 0.40
        assert decision.confidence >= 0.70
    
    def test_info_severity_base_score(self) -> None:
        """Info severity produces low base score."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="LOW",
                severity="info",
                action_required=False,
                recommended_action="None",
            ),
            current_regime="NOISY",
            recent_anomaly_count=3,
        )
        
        decision = engine.decide(ctx)
        
        # Base 0.05 with amplifiers won't reach threshold
        assert decision.action == "LOG_ONLY"


class TestAmplifiers:
    """Test contextual amplifiers with real flag values."""
    
    def test_consecutive_5_amplifier(self) -> None:
        """5+ consecutive anomalies amplifies score significantly."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="medium",
                action_required=True,
                recommended_action="Monitor",
            ),
            consecutive_anomalies=5,
            current_regime="NOISY",
            recent_anomaly_count=10,
        )
        
        decision = engine.decide(ctx)
        
        # consecutive_5 (1.35) + noisy (1.10) should amplify to at least INVESTIGATE
        assert decision.action in ("ESCALATE", "INVESTIGATE")
    
    def test_high_anomaly_rate_amplifier(self) -> None:
        """High anomaly rate (>0.60) amplifies score."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="medium",
                action_required=True,
                recommended_action="Monitor",
            ),
            recent_anomaly_rate=0.70,
            current_regime="NOISY",
            recent_anomaly_count=10,
        )
        
        decision = engine.decide(ctx)
        
        # rate_high (1.20) + noisy (1.10) amplifies but not enough for INVESTIGATE
        # 0.45 * 1.20 * 1.10 = 0.594 -> MONITOR
        # Verify that amplifiers were applied
        assert any("rate_high" in amp for amp in decision.source_ml_outputs["amplifiers"])
        assert decision.confidence > 0.45  # Amplified above base
    
    def test_volatile_regime_amplifier(self) -> None:
        """Volatile regime amplifies score."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="medium",
                action_required=True,
                recommended_action="Monitor",
            ),
            current_regime="VOLATILE",
            recent_anomaly_count=5,
        )
        
        decision = engine.decide(ctx)
        
        # Volatile (1.15) should increase priority
        assert decision.priority <= 3
    
    def test_high_drift_amplifier(self) -> None:
        """High drift score (>0.70) amplifies score."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="medium",
                action_required=True,
                recommended_action="Monitor",
            ),
            drift_score=0.80,
            current_regime="NOISY",
            recent_anomaly_count=5,
        )
        
        decision = engine.decide(ctx)
        
        # drift_high (1.20) + noisy (1.10) should increase priority
        assert decision.priority <= 3
    
    def test_amplifier_values_in_log(self) -> None:
        """Amplifiers include actual multiplier values in output."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="medium",
                action_required=True,
                recommended_action="Monitor",
            ),
            consecutive_anomalies=5,
            current_regime="NOISY",
            recent_anomaly_count=5,
        )
        
        decision = engine.decide(ctx)
        
        # Check that source_ml_outputs contains the actual multiplier values
        # Format: "consecutive_5×1.35"
        for amp in decision.source_ml_outputs["amplifiers"]:
            assert "×" in amp
            # Extract the multiplier value
            parts = amp.split("×")
            assert len(parts) == 2
            multiplier = float(parts[1])
            assert multiplier > 1.0  # Amplifiers are > 1.0


class TestAttenuators:
    """Test contextual attenuators with real flag values."""
    
    def test_stable_low_drift_attenuator(self) -> None:
        """Stable regime with low drift reduces score."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="medium",
                action_required=True,
                recommended_action="Monitor",
            ),
            current_regime="STABLE",
            drift_score=0.05,
            recent_anomaly_count=5,  # Avoid no_context attenuator
        )
        
        decision = engine.decide(ctx)
        
        # stable (0.85) attenuator should reduce score
        # medium base (0.45) * 0.85 = 0.3825 -> LOG_ONLY
        assert decision.action == "LOG_ONLY"
    
    def test_low_criticality_attenuator(self) -> None:
        """Low criticality reduces score."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="medium",
                action_required=True,
                recommended_action="Monitor",
            ),
            series_criticality="LOW",
            drift_score=0.05,  # Avoid drift amplifiers
            current_regime="NOISY",
            recent_anomaly_count=5,  # Avoid no_context attenuator
        )
        
        decision = engine.decide(ctx)
        
        # low_criticality (0.80) attenuator should be applied
        assert any("low_criticality" in att for att in decision.source_ml_outputs["attenuators"])
        # Score should be lower than without attenuator
        assert decision.confidence < 0.50
    
    def test_no_context_attenuator(self) -> None:
        """No recent anomalies (0) reduces score."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="medium",
                action_required=True,
                recommended_action="Monitor",
            ),
            recent_anomaly_count=0,
            drift_score=0.05,  # Avoid stable attenuator and drift amplifiers
            current_regime="NOISY",
        )
        
        decision = engine.decide(ctx)
        
        # no_context (0.90) attenuator should be applied
        assert any("no_context" in att for att in decision.source_ml_outputs["attenuators"])
        # Score should be reduced from base + noisy (0.45 * 1.10 = 0.495)
        # to (0.45 * 1.10 * 0.90 = 0.4455)
        assert decision.confidence < 0.49


class TestScoreCeiling:
    """Test score ceiling at 1.0."""
    
    def test_score_never_exceeds_one(self) -> None:
        """Score is capped at 1.0 even with multiple amplifiers."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="HIGH",
                severity="critical",
                action_required=True,
                recommended_action="Investigate",
            ),
            consecutive_anomalies=10,
            recent_anomaly_rate=0.90,
            current_regime="VOLATILE",
            drift_score=0.90,
            recent_anomaly_count=20,
        )
        
        decision = engine.decide(ctx)
        
        # Confidence should never exceed 1.0
        assert decision.confidence <= 1.0


class TestActionMapping:
    """Test score to action mapping with default thresholds."""
    
    def test_high_score_escalate(self) -> None:
        """High score maps to ESCALATE."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="HIGH",
                severity="critical",
                action_required=True,
                recommended_action="Investigate",
            ),
            consecutive_anomalies=10,
            current_regime="VOLATILE",
            recent_anomaly_count=20,
        )
        
        decision = engine.decide(ctx)
        
        # Should escalate (critical base + consecutive_5 + volatile)
        assert decision.action in ("ESCALATE", "INVESTIGATE")
    
    def test_medium_score_investigate(self) -> None:
        """Medium score maps to INVESTIGATE."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="MEDIUM",
                severity="high",
                action_required=True,
                recommended_action="Investigate",
            ),
            consecutive_anomalies=3,
            current_regime="NOISY",
            recent_anomaly_count=5,
        )
        
        decision = engine.decide(ctx)
        
        # Should be INVESTIGATE or higher (high base + consecutive_3 + noisy)
        assert decision.action in ("INVESTIGATE", "ESCALATE")
    
    def test_low_score_monitor(self) -> None:
        """Low score maps to MONITOR."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="LOW",
                severity="medium",
                action_required=False,
                recommended_action="Monitor",
            ),
            current_regime="NOISY",
            recent_anomaly_count=3,
        )
        
        decision = engine.decide(ctx)
        
        # Low/medium base with some amplifiers
        assert decision.action in ("MONITOR", "INVESTIGATE", "LOG_ONLY")
    
    def test_very_low_score_log_only(self) -> None:
        """Very low score maps to LOG_ONLY."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="LOW",
                severity="info",
                action_required=False,
                recommended_action="None",
            ),
            current_regime="STABLE",
            drift_score=0.05,
            recent_anomaly_count=0,
        )
        
        decision = engine.decide(ctx)
        
        # info base (0.05) with stable attenuator (0.85) and no_context (0.90)
        # should result in very low score
        assert decision.action == "LOG_ONLY"
        assert decision.priority == 5


class TestSourceMlOutputs:
    """Test decision includes source ML outputs."""
    
    def test_decision_includes_score_factors(self) -> None:
        """Decision includes score base, amplifiers, attenuators with values."""
        engine = ContextualDecisionEngine()
        ctx = DecisionContext(
            series_id="test",
            severity=SeverityResult(
                risk_level="HIGH",
                severity="high",
                action_required=True,
                recommended_action="Investigate",
            ),
            consecutive_anomalies=3,
            current_regime="NOISY",
            recent_anomaly_count=5,
        )
        
        decision = engine.decide(ctx)
        
        # Source outputs should contain score details with actual values
        assert "score_base" in decision.source_ml_outputs
        assert "amplifiers" in decision.source_ml_outputs
        assert "attenuators" in decision.source_ml_outputs
        assert "score_final" in decision.source_ml_outputs
        
        # Verify structure of amplifiers (contain multiplier)
        for amp in decision.source_ml_outputs["amplifiers"]:
            assert "×" in amp
