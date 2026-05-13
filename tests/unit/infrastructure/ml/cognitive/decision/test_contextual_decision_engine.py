"""Tests for ContextualDecisionEngine with config injection.

Verifies:
- Config defaults produce expected behavior
- Config validation rejects invalid ranges
- Amplifiers/attenuators never violate bounds [0, 1]
- Exclusive categories work correctly
- Config injection overrides defaults
- WARNING = MEDIUM is intentional

All tests use real engine behavior with injected configs (no mocks).
"""

from __future__ import annotations

import dataclasses
import re
from typing import Optional

import pytest

from iot_machine_learning.domain.entities.anomaly import AnomalyResult, AnomalySeverity
from iot_machine_learning.domain.entities.decision import DecisionContext
from iot_machine_learning.infrastructure.ml.cognitive.decision.contextual_decision_config import (
    ContextualDecisionConfig,
)
from iot_machine_learning.infrastructure.ml.cognitive.decision.contextual_decision_engine import (
    ContextualDecisionEngine,
)


# ========== HELPER FUNCTIONS ==========


def build_test_context(
    severity: str = "NONE",
    consecutive: int = 0,
    anomaly_rate: float = 0.0,
    regime: str = "STABLE",
    drift_score: float = 0.0,
    criticality: str = "MEDIUM",
    recent_anomaly_count: int = 0,
    series_id: str = "test_series",
) -> DecisionContext:
    """Build a test DecisionContext with sensible defaults.
    
    Args:
        severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW, NONE, WARNING).
        consecutive: Number of consecutive anomalies.
        anomaly_rate: Recent anomaly rate [0.0, 1.0].
        regime: Current regime (STABLE, VOLATILE, NOISY, TRENDING).
        drift_score: Drift score [0.0, 1.0].
        criticality: Series criticality (LOW, MEDIUM, HIGH).
        recent_anomaly_count: Count of recent anomalies.
        series_id: Series identifier.
    
    Returns:
        DecisionContext for testing.
    """
    # Map severity string to AnomalySeverity enum
    severity_map = {
        "CRITICAL": AnomalySeverity.CRITICAL,
        "HIGH": AnomalySeverity.HIGH,
        "MEDIUM": AnomalySeverity.MEDIUM,
        "LOW": AnomalySeverity.LOW,
        "NONE": AnomalySeverity.NONE,
        "WARNING": AnomalySeverity.MEDIUM,  # WARNING maps to MEDIUM
    }
    severity_enum = severity_map.get(severity.upper(), AnomalySeverity.NONE)
    
    # Build simple severity object for DecisionContext
    # DecisionContext expects an object with .severity attribute
    class SimpleSeverity:
        def __init__(self, sev: str):
            self.severity = sev
    
    return DecisionContext(
        series_id=series_id,
        severity=SimpleSeverity(severity),
        consecutive_anomalies=consecutive,
        recent_anomaly_rate=anomaly_rate,
        current_regime=regime,
        drift_score=drift_score,
        series_criticality=criticality,
        recent_anomaly_count=recent_anomaly_count,
        audit_trace_id="test_trace",
    )


# ========== TESTS ==========


class TestContextualDecisionEngineConfig:
    """Test suite for ContextualDecisionEngine configuration and scoring."""
    
    def test_config_defaults_produce_expected_scores(self):
        """Verify that default config produces scores in correct ranges.
        
        WHY: Ensures default config is sane and produces expected behavior.
        
        Validates:
        - CRITICAL without amplifiers → score >= threshold_escalate
        - HIGH without amplifiers → score in [threshold_investigate, threshold_escalate)
        - LOW without amplifiers → score < threshold_investigate
        - Relative order: critical > high > medium > low > none
        """
        engine = ContextualDecisionEngine()
        cfg = engine._config
        
        # Test CRITICAL severity (no amplifiers/attenuators)
        ctx_critical = build_test_context(
            severity="CRITICAL",
            consecutive=0,
            anomaly_rate=0.0,
            regime="STABLE",
            drift_score=0.0,
            criticality="MEDIUM",
            recent_anomaly_count=1,  # Has context to avoid no_context attenuator
        )
        result_critical = engine.decide(ctx_critical)
        
        # CRITICAL should be >= escalate threshold
        assert result_critical.confidence >= cfg.threshold_escalate, (
            f"CRITICAL score {result_critical.confidence} should be >= "
            f"threshold_escalate {cfg.threshold_escalate}"
        )
        
        # Test HIGH severity
        ctx_high = build_test_context(
            severity="HIGH",
            consecutive=0,
            anomaly_rate=0.0,
            regime="STABLE",
            drift_score=0.0,
            criticality="MEDIUM",
            recent_anomaly_count=1,
        )
        result_high = engine.decide(ctx_high)
        
        # HIGH should be in [investigate, escalate)
        assert cfg.threshold_investigate <= result_high.confidence < cfg.threshold_escalate, (
            f"HIGH score {result_high.confidence} should be in "
            f"[{cfg.threshold_investigate}, {cfg.threshold_escalate})"
        )
        
        # Test LOW severity
        ctx_low = build_test_context(
            severity="LOW",
            consecutive=0,
            anomaly_rate=0.0,
            regime="STABLE",
            drift_score=0.0,
            criticality="MEDIUM",
            recent_anomaly_count=1,
        )
        result_low = engine.decide(ctx_low)
        
        # LOW should be < investigate threshold
        assert result_low.confidence < cfg.threshold_investigate, (
            f"LOW score {result_low.confidence} should be < "
            f"threshold_investigate {cfg.threshold_investigate}"
        )
        
        # Test relative ordering (without amplifiers/attenuators)
        ctx_none = build_test_context(severity="NONE", recent_anomaly_count=1)
        ctx_medium = build_test_context(severity="MEDIUM", recent_anomaly_count=1)
        
        result_none = engine.decide(ctx_none)
        result_medium = engine.decide(ctx_medium)
        
        # Verify strict ordering
        assert result_none.confidence < result_low.confidence, "none < low"
        assert result_low.confidence < result_medium.confidence, "low < medium"
        assert result_medium.confidence < result_high.confidence, "medium < high"
        assert result_high.confidence < result_critical.confidence, "high < critical"
    
    @pytest.mark.parametrize("field,invalid_value,expected_error", [
        ("score_critical", 1.5, "score_critical=1.5 must be in [0.0, 1.0]"),
        ("score_critical", -0.1, "score_critical=-0.1 must be in [0.0, 1.0]"),
        ("score_high", 1.1, "score_high=1.1 must be in [0.0, 1.0]"),
        ("amp_consecutive_5", -0.1, "amp_consecutive_5=-0.1 must be > 0"),
        ("att_stable", -0.5, "att_stable=-0.5 must be > 0"),
        ("flag_cache_ttl_seconds", 5, "flag_cache_ttl_seconds=5 must be in [10, 300]"),
        ("flag_cache_ttl_seconds", 400, "flag_cache_ttl_seconds=400 must be in [10, 300]"),
    ])
    def test_config_validate_rejects_invalid_ranges(
        self, field: str, invalid_value: float, expected_error: str
    ):
        """Verify that config validation rejects invalid parameter values.
        
        WHY: Prevents misconfiguration that could lead to incorrect decisions.
        
        Tests validation of:
        - Scores must be in [0.0, 1.0]
        - Amplifiers must be > 0
        - Attenuators must be > 0
        - TTL must be in [10, 300]
        """
        # Create config with invalid value using dataclasses.replace
        config_invalid = dataclasses.replace(
            ContextualDecisionConfig(),
            **{field: invalid_value}
        )
        
        # Validation should raise ValueError mentioning the field
        with pytest.raises(ValueError, match=re.escape(expected_error)):
            config_invalid.validate()
    
    def test_threshold_order_validated(self):
        """Verify that threshold ordering is validated.
        
        WHY: Thresholds must be strictly ordered for correct action mapping.
        
        Tests:
        - escalate > investigate > monitor > 0
        - Reversed order is rejected
        """
        # Test: escalate <= investigate (invalid)
        config_bad_1 = dataclasses.replace(
            ContextualDecisionConfig(),
            threshold_escalate=0.50,
            threshold_investigate=0.75,
        )
        with pytest.raises(ValueError, match="Thresholds must be strictly increasing"):
            config_bad_1.validate()
        
        # Test: investigate <= monitor (invalid)
        config_bad_2 = dataclasses.replace(
            ContextualDecisionConfig(),
            threshold_investigate=0.30,
            threshold_monitor=0.50,
        )
        with pytest.raises(ValueError, match="Thresholds must be strictly increasing"):
            config_bad_2.validate()
        
        # Test: monitor <= 0 (invalid)
        config_bad_3 = dataclasses.replace(
            ContextualDecisionConfig(),
            threshold_monitor=0.0,
        )
        with pytest.raises(ValueError, match="Thresholds must be strictly increasing"):
            config_bad_3.validate()
        
        # Test: valid ordering (should not raise)
        config_valid = dataclasses.replace(
            ContextualDecisionConfig(),
            threshold_escalate=0.80,
            threshold_investigate=0.60,
            threshold_monitor=0.30,
        )
        config_valid.validate()  # Should not raise
    
    def test_amplifiers_never_exceed_1(self):
        """Verify that amplifiers never push score above 1.0.
        
        WHY: Score must remain in [0, 1] range for valid probability interpretation.
        
        Tests extreme case where all amplifiers are active simultaneously
        with exaggerated values.
        """
        # Config with exaggerated amplifiers to force potential overflow
        config_extreme = dataclasses.replace(
            ContextualDecisionConfig(),
            score_critical=0.90,  # High base score
            amp_consecutive_5=0.50,  # Exaggerated
            amp_rate_high=0.50,
            amp_volatile=0.50,
            amp_drift_high=0.50,
        )
        
        engine = ContextualDecisionEngine(config=config_extreme)
        
        # Context that activates ALL amplifiers
        ctx_all_amplifiers = build_test_context(
            severity="CRITICAL",
            consecutive=10,  # >= 5 (activates consecutive_5)
            anomaly_rate=0.80,  # > 0.60 (activates rate_high)
            regime="VOLATILE",  # Activates volatile
            drift_score=0.90,  # > 0.70 (activates drift_high)
            criticality="HIGH",  # No attenuators
            recent_anomaly_count=5,  # Has context
        )
        
        result = engine.decide(ctx_all_amplifiers)
        
        # Score must never exceed 1.0
        assert result.confidence <= 1.0, (
            f"Score {result.confidence} exceeds maximum 1.0 even with all amplifiers active"
        )
        
        # Score should be exactly 1.0 (clamped)
        assert result.confidence == 1.0, (
            f"With extreme amplifiers, score should be clamped to 1.0, got {result.confidence}"
        )
    
    def test_attenuators_never_go_negative(self):
        """Verify that attenuators never push score below 0.0.
        
        WHY: Score must remain in [0, 1] range. Negative scores are invalid.
        
        Tests extreme case where all attenuators are active simultaneously
        with low base score and exaggerated attenuation values.
        """
        # Config with low base score and exaggerated attenuators
        config_extreme = dataclasses.replace(
            ContextualDecisionConfig(),
            score_none=0.05,  # Minimal base score
            att_stable=0.40,  # Exaggerated (would subtract 0.40)
            att_low_criticality=0.40,  # Exaggerated
            att_no_context=0.40,  # Exaggerated
            att_stable_drift_max=0.50,  # High threshold to activate stable
        )
        
        engine = ContextualDecisionEngine(config=config_extreme)
        
        # Context that activates ALL attenuators
        ctx_all_attenuators = build_test_context(
            severity="NONE",  # Minimal base score (0.05)
            consecutive=0,
            anomaly_rate=0.0,
            regime="STABLE",  # Activates stable attenuator
            drift_score=0.05,  # < 0.50 (activates stable)
            criticality="LOW",  # Activates low_criticality
            recent_anomaly_count=0,  # Activates no_context
        )
        
        result = engine.decide(ctx_all_attenuators)
        
        # Score must never go negative
        assert result.confidence >= 0.0, (
            f"Score {result.confidence} is negative even with all attenuators active"
        )
        
        # Score should be exactly 0.0 (floored)
        assert result.confidence == 0.0, (
            f"With extreme attenuators, score should be floored to 0.0, got {result.confidence}"
        )
    
    def test_exclusive_categories_only_apply_highest_priority(self):
        """Verify that exclusive categories only apply highest priority rule.
        
        WHY: Within each category (consecutive, rate, regime, drift), only
        the highest-priority matching rule should apply to avoid double-counting.
        
        Tests:
        - consecutive=5 applies amp_consecutive_5 (not both)
        - consecutive=3 applies amp_consecutive_3 (not both)
        - Difference matches expected delta
        """
        engine = ContextualDecisionEngine()
        cfg = engine._config
        base_score = cfg.score_medium  # Use medium as base
        
        # Context with consecutive=5 (activates both rules, but only first applies)
        ctx_5 = build_test_context(
            severity="MEDIUM",
            consecutive=5,
            anomaly_rate=0.0,
            regime="STABLE",
            drift_score=0.0,
            criticality="MEDIUM",
            recent_anomaly_count=1,
        )
        
        # Context with consecutive=3 (activates only second rule)
        ctx_3 = build_test_context(
            severity="MEDIUM",
            consecutive=3,
            anomaly_rate=0.0,
            regime="STABLE",
            drift_score=0.0,
            criticality="MEDIUM",
            recent_anomaly_count=1,
        )
        
        result_5 = engine.decide(ctx_5)
        result_3 = engine.decide(ctx_3)
        
        # Expected difference: amp_consecutive_5 - amp_consecutive_3
        expected_diff = cfg.amp_consecutive_5 - cfg.amp_consecutive_3
        actual_diff = result_5.confidence - result_3.confidence
        
        # Verify exclusive behavior (only one amplifier per category)
        assert abs(actual_diff - expected_diff) < 1e-9, (
            f"Exclusive category failed: consecutive=5 should apply only amp_consecutive_5. "
            f"Expected diff {expected_diff}, got {actual_diff}"
        )
        
        # Verify that consecutive=5 score is higher than consecutive=3
        assert result_5.confidence > result_3.confidence, (
            "consecutive=5 should have higher score than consecutive=3"
        )
    
    def test_config_injection_overrides_defaults(self):
        """Verify that engine uses injected config, not hardcoded defaults.
        
        WHY: Ensures DIP is properly implemented - engine depends on config
        abstraction, not on hardcoded values.
        
        Tests:
        - Custom score_critical is used
        - Custom threshold_escalate is used
        - No fallback to hardcoded defaults
        """
        # Custom config with unusual values (impossible to get by coincidence)
        custom_config = dataclasses.replace(
            ContextualDecisionConfig(),
            score_critical=0.99,  # Unusual value
            threshold_escalate=0.98,  # Just below score_critical
            threshold_investigate=0.60,
            threshold_monitor=0.30,
        )
        
        engine = ContextualDecisionEngine(config=custom_config)
        
        # Context with CRITICAL severity, no amplifiers/attenuators
        # Use TRENDING regime to avoid stable attenuator
        ctx_critical = build_test_context(
            severity="CRITICAL",
            consecutive=0,
            anomaly_rate=0.0,
            regime="TRENDING",  # Not STABLE, so no attenuator
            drift_score=0.0,
            criticality="MEDIUM",
            recent_anomaly_count=1,
        )
        
        result = engine.decide(ctx_critical)
        
        # Score should be exactly 0.99 (custom score_critical)
        assert abs(result.confidence - 0.99) < 1e-9, (
            f"Engine should use injected score_critical=0.99, got {result.confidence}"
        )
        
        # Action should be ESCALATE (score >= custom threshold_escalate)
        assert result.action == "ESCALATE", (
            f"With score {result.confidence} >= threshold_escalate {custom_config.threshold_escalate}, "
            f"action should be ESCALATE, got {result.action}"
        )
        
        # Verify engine is using custom config, not defaults
        assert engine._config.score_critical == 0.99, "Config injection failed"
        assert engine._config.threshold_escalate == 0.98, "Config injection failed"
    
    def test_warning_equals_medium_is_intentional(self):
        """Document and verify that WARNING == MEDIUM is intentional behavior.
        
        WHY: Audit identified WARNING = MEDIUM as potential bug. This test
        documents that it's intentional in this domain.
        
        If future requirements differentiate WARNING from MEDIUM, this test
        will fail and serve as a reminder to update the config.
        """
        config = ContextualDecisionConfig()
        
        assert config.score_warning == config.score_medium, (
            "WARNING == MEDIUM is intentional behavior in this domain. "
            "WARNING is treated as an alias of MEDIUM severity. "
            "If this changes in the future, update ContextualDecisionConfig.score_warning "
            "and this test."
        )
        
        # Verify that both produce the same score
        engine = ContextualDecisionEngine()
        
        ctx_warning = build_test_context(severity="WARNING", recent_anomaly_count=1)
        ctx_medium = build_test_context(severity="MEDIUM", recent_anomaly_count=1)
        
        result_warning = engine.decide(ctx_warning)
        result_medium = engine.decide(ctx_medium)
        
        assert result_warning.confidence == result_medium.confidence, (
            f"WARNING and MEDIUM should produce identical scores, "
            f"got {result_warning.confidence} vs {result_medium.confidence}"
        )
    
    def test_amplifier_threshold_ordering_validated(self):
        """Verify that amplifier threshold pairs are validated.
        
        WHY: Ensures high thresholds are > medium thresholds within each category.
        
        Tests:
        - amp_consecutive_high_count > amp_consecutive_med_count
        - amp_rate_high_threshold > amp_rate_med_threshold
        - amp_drift_high_threshold > amp_drift_med_threshold
        """
        # Test: consecutive thresholds reversed
        config_bad_consecutive = dataclasses.replace(
            ContextualDecisionConfig(),
            amp_consecutive_high_count=3,
            amp_consecutive_med_count=5,
        )
        with pytest.raises(ValueError, match="amp_consecutive_med_count"):
            config_bad_consecutive.validate()
        
        # Test: rate thresholds reversed
        config_bad_rate = dataclasses.replace(
            ContextualDecisionConfig(),
            amp_rate_high_threshold=0.30,
            amp_rate_med_threshold=0.60,
        )
        with pytest.raises(ValueError, match="amp_rate_med_threshold"):
            config_bad_rate.validate()
        
        # Test: drift thresholds reversed
        config_bad_drift = dataclasses.replace(
            ContextualDecisionConfig(),
            amp_drift_high_threshold=0.40,
            amp_drift_med_threshold=0.70,
        )
        with pytest.raises(ValueError, match="amp_drift_med_threshold"):
            config_bad_drift.validate()
    
    def test_priority_gap_detection(self):
        """Verify that priority gaps are detected during validation.
        
        WHY: Audit found original gap (1,2,3,5). Config should validate
        that priorities are consecutive.
        
        NOTE: This test validates the config dataclass itself.
        The engine's _build_amplifiers() uses hardcoded priorities,
        so this test ensures the config's priority fields are consistent.
        """
        config = ContextualDecisionConfig()
        
        # Verify priorities are consecutive (1,2,3,4)
        priorities = [
            config.priority_escalate,
            config.priority_investigate,
            config.priority_monitor,
            config.priority_log_only,
        ]
        
        # Check no gaps
        for i in range(len(priorities) - 1):
            gap = priorities[i + 1] - priorities[i]
            assert gap == 1, (
                f"Priority gap detected: {priorities[i]} → {priorities[i + 1]} "
                f"(gap of {gap}, should be 1)"
            )
        
        # Verify validation catches gaps
        config_with_gap = dataclasses.replace(
            ContextualDecisionConfig(),
            priority_escalate=1,
            priority_investigate=2,
            priority_monitor=3,
            priority_log_only=5,  # Gap!
        )
        
        with pytest.raises(ValueError, match="Priority gap detected"):
            config_with_gap.validate()
