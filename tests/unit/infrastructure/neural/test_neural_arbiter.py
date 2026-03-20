"""Tests for neural arbiter."""

import pytest
from infrastructure.ml.cognitive.neural.competition.arbiter import NeuralArbiter
from infrastructure.ml.cognitive.neural.competition.confidence_comparator import ConfidenceComparator
from infrastructure.ml.cognitive.neural.competition.outcome_tracker import OutcomeTracker
from infrastructure.ml.cognitive.neural.types import NeuralResult, InputType
from infrastructure.ml.cognitive.universal.analysis.types import UniversalResult
from iot_machine_learning.domain.entities.explainability import Explanation
from iot_machine_learning.domain.services.severity_rules import SeverityResult


class TestArbiterInitialization:
    """Test arbiter initialization."""
    
    def test_default_initialization(self):
        """Test arbiter with default components."""
        arbiter = NeuralArbiter()
        
        assert arbiter.comparator is not None
        assert arbiter.tracker is not None
        assert arbiter.history_weight == 0.1
    
    def test_custom_components(self):
        """Test arbiter with custom components."""
        comparator = ConfidenceComparator(neural_margin=0.15)
        tracker = OutcomeTracker()
        
        arbiter = NeuralArbiter(
            confidence_comparator=comparator,
            outcome_tracker=tracker,
            history_weight=0.2,
        )
        
        assert arbiter.comparator.neural_margin == 0.15
        assert arbiter.history_weight == 0.2


class TestConfidenceBasedDecision:
    """Test primary decision factor: confidence."""
    
    def test_neural_wins_with_higher_confidence(self):
        """Test neural wins when confidence is higher."""
        arbiter = NeuralArbiter()
        
        neural_result = _create_neural_result(confidence=0.9)
        universal_result = _create_universal_result(confidence=0.7)
        
        winner, confidence, reason = arbiter.arbitrate(
            neural_result, universal_result, "test_domain"
        )
        
        assert winner == "neural"
        assert confidence == 0.9
    
    def test_universal_wins_with_higher_confidence(self):
        """Test universal wins when confidence is higher."""
        arbiter = NeuralArbiter()
        
        neural_result = _create_neural_result(confidence=0.6)
        universal_result = _create_universal_result(confidence=0.85)
        
        winner, confidence, reason = arbiter.arbitrate(
            neural_result, universal_result, "test_domain"
        )
        
        assert winner == "universal"
        assert confidence == 0.85
    
    def test_universal_wins_within_margin(self):
        """Test universal wins when neural within margin."""
        arbiter = NeuralArbiter(
            confidence_comparator=ConfidenceComparator(neural_margin=0.1)
        )
        
        neural_result = _create_neural_result(confidence=0.75)
        universal_result = _create_universal_result(confidence=0.70)
        
        winner, confidence, reason = arbiter.arbitrate(
            neural_result, universal_result, "test_domain"
        )
        
        # Neural only 0.05 ahead, needs 0.1
        assert winner == "universal"


class TestMonteCarloConsistency:
    """Test secondary factor: Monte Carlo consistency."""
    
    def test_monte_carlo_influences_decision(self):
        """Test Monte Carlo uncertainty affects decision."""
        arbiter = NeuralArbiter()
        
        # Create results with Monte Carlo
        neural_mc = type('MC', (), {
            'uncertainty_class': 'low',
            'ci_width': 0.2,
        })()
        
        universal_mc = type('MC', (), {
            'uncertainty_class': 'high',
            'ci_width': 0.5,
        })()
        
        neural_result = _create_neural_result(confidence=0.8, monte_carlo=neural_mc)
        universal_result = _create_universal_result(confidence=0.78, monte_carlo=universal_mc)
        
        winner, confidence, reason = arbiter.arbitrate(
            neural_result, universal_result, "test_domain"
        )
        
        # Neural has higher confidence and lower uncertainty
        assert winner == "neural"


class TestDomainHistory:
    """Test tertiary factor: domain win history."""
    
    def test_history_tracked(self):
        """Test win history is recorded."""
        arbiter = NeuralArbiter()
        
        neural_result = _create_neural_result(confidence=0.9)
        universal_result = _create_universal_result(confidence=0.7)
        
        # First decision
        arbiter.arbitrate(neural_result, universal_result, "domain_1")
        
        # Check history
        stats = arbiter.get_win_statistics("domain_1")
        
        assert stats["neural_wins"] == 1
        assert stats["universal_wins"] == 0
    
    def test_multiple_domains_tracked_separately(self):
        """Test different domains tracked independently."""
        arbiter = NeuralArbiter()
        
        neural_high = _create_neural_result(confidence=0.9)
        universal_high = _create_universal_result(confidence=0.9)
        
        # Domain 1: neural wins
        arbiter.arbitrate(neural_high, _create_universal_result(0.6), "domain_1")
        
        # Domain 2: universal wins
        arbiter.arbitrate(_create_neural_result(0.6), universal_high, "domain_2")
        
        stats1 = arbiter.get_win_statistics("domain_1")
        stats2 = arbiter.get_win_statistics("domain_2")
        
        assert stats1["neural_wins"] == 1
        assert stats2["universal_wins"] == 1
    
    def test_preferred_engine_emerges(self):
        """Test preferred engine identified after enough decisions."""
        arbiter = NeuralArbiter()
        
        neural_high = _create_neural_result(confidence=0.9)
        universal_low = _create_universal_result(confidence=0.6)
        
        # Neural wins 12 times
        for _ in range(12):
            arbiter.arbitrate(neural_high, universal_low, "neural_domain")
        
        stats = arbiter.get_win_statistics("neural_domain")
        
        assert stats["preferred_engine"] == "neural"


class TestWinStatistics:
    """Test win statistics reporting."""
    
    def test_get_win_statistics(self):
        """Test statistics structure."""
        arbiter = NeuralArbiter()
        
        neural_result = _create_neural_result(confidence=0.85)
        universal_result = _create_universal_result(confidence=0.75)
        
        arbiter.arbitrate(neural_result, universal_result, "test_domain")
        
        stats = arbiter.get_win_statistics("test_domain")
        
        assert "neural_wins" in stats
        assert "universal_wins" in stats
        assert "total_decisions" in stats
        assert "neural_rate" in stats
        assert "universal_rate" in stats
        assert "preferred_engine" in stats
    
    def test_statistics_for_unknown_domain(self):
        """Test statistics for domain with no history."""
        arbiter = NeuralArbiter()
        
        stats = arbiter.get_win_statistics("unknown_domain")
        
        assert stats["total_decisions"] == 0
        assert stats["neural_rate"] == 0.5


class TestReasonReporting:
    """Test arbitration reason strings."""
    
    def test_reason_includes_confidence(self):
        """Test reason includes confidence values."""
        arbiter = NeuralArbiter()
        
        neural_result = _create_neural_result(confidence=0.9)
        universal_result = _create_universal_result(confidence=0.7)
        
        _, _, reason = arbiter.arbitrate(
            neural_result, universal_result, "test_domain"
        )
        
        assert "0.9" in reason or "0.90" in reason
    
    def test_reason_includes_margin(self):
        """Test reason includes margin information."""
        arbiter = NeuralArbiter()
        
        neural_result = _create_neural_result(confidence=0.85)
        universal_result = _create_universal_result(confidence=0.70)
        
        _, _, reason = arbiter.arbitrate(
            neural_result, universal_result, "test_domain"
        )
        
        assert "margin" in reason.lower() or "neural" in reason.lower()


class TestEdgeCases:
    """Test edge cases."""
    
    def test_equal_confidence(self):
        """Test decision when confidences are equal."""
        arbiter = NeuralArbiter()
        
        neural_result = _create_neural_result(confidence=0.8)
        universal_result = _create_universal_result(confidence=0.8)
        
        winner, confidence, reason = arbiter.arbitrate(
            neural_result, universal_result, "test_domain"
        )
        
        # Universal should win (neural needs margin)
        assert winner == "universal"
        assert confidence == 0.8
    
    def test_very_low_confidence_both(self):
        """Test both engines have low confidence."""
        arbiter = NeuralArbiter()
        
        neural_result = _create_neural_result(confidence=0.3)
        universal_result = _create_universal_result(confidence=0.35)
        
        winner, confidence, reason = arbiter.arbitrate(
            neural_result, universal_result, "test_domain"
        )
        
        # Universal wins (higher confidence, even if low)
        assert winner == "universal"
    
    def test_very_high_confidence_both(self):
        """Test both engines have high confidence."""
        arbiter = NeuralArbiter()
        
        neural_result = _create_neural_result(confidence=0.95)
        universal_result = _create_universal_result(confidence=0.93)
        
        winner, confidence, reason = arbiter.arbitrate(
            neural_result, universal_result, "test_domain"
        )
        
        # Universal wins (neural doesn't exceed margin)
        assert winner == "universal"


# Helper functions

def _create_neural_result(confidence: float, monte_carlo=None) -> NeuralResult:
    """Create mock NeuralResult."""
    return NeuralResult(
        severity="medium",
        confidence=confidence,
        spike_patterns={},
        firing_rates={},
        energy_consumed=1e-10,
        active_neurons=10,
        silent_neurons=5,
        domain="test",
        input_type=InputType.TEXT,
        monte_carlo=monte_carlo,
    )


def _create_universal_result(confidence: float, monte_carlo=None) -> UniversalResult:
    """Create mock UniversalResult."""
    explanation = Explanation(
        summary_severity="medium",
        summary_value=0.5,
        summary_confidence=confidence,
        summary_trend="stable",
    )
    
    severity = SeverityResult(
        risk_level="MEDIUM",
        severity="medium",
        action_required=False,
        recommended_action="Monitor",
    )
    
    return UniversalResult(
        explanation=explanation,
        severity=severity,
        analysis={},
        confidence=confidence,
        domain="test",
        input_type=InputType.TEXT,
        monte_carlo=monte_carlo,
    )
