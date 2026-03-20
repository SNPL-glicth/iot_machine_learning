"""Tests for Monte Carlo simulation engine."""

from __future__ import annotations

import pytest
import time
from datetime import datetime

from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.monte_carlo import (
    MonteCarloResult,
    MonteCarloSimulator,
)
from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.types import InputType


class TestMonteCarloResult:
    """Test MonteCarloResult dataclass."""

    def test_monte_carlo_result_creation(self) -> None:
        """MonteCarloResult should be creatable."""
        result = MonteCarloResult(
            n_simulations=1000,
            severity_distribution={"critical": 0.73, "warning": 0.21, "info": 0.06},
            confidence_interval=(0.65, 0.81),
            expected_severity="critical",
            confidence_score=0.73,
            scenario_outcomes={"best_case": {}, "worst_case": {}, "most_likely": {}},
            uncertainty_level="low",
        )
        
        assert result.n_simulations == 1000
        assert result.expected_severity == "critical"
        assert result.confidence_score == 0.73

    def test_monte_carlo_result_to_dict(self) -> None:
        """MonteCarloResult should serialize to dict."""
        result = MonteCarloResult(
            n_simulations=500,
            severity_distribution={"warning": 0.6, "info": 0.4},
            confidence_interval=(0.3, 0.7),
            expected_severity="warning",
            confidence_score=0.6,
            scenario_outcomes={},
            uncertainty_level="moderate",
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["n_simulations"] == 500
        assert result_dict["expected_severity"] == "warning"
        assert "confidence_interval" in result_dict


class TestMonteCarloSimulator:
    """Test MonteCarloSimulator."""

    def test_simulator_instantiation(self) -> None:
        """Simulator should instantiate without errors."""
        sim = MonteCarloSimulator(n_simulations=1000)
        assert sim is not None

    def test_simulator_clamps_n_simulations(self) -> None:
        """Should clamp n_simulations to valid range."""
        sim_low = MonteCarloSimulator(n_simulations=50)
        assert sim_low._n_simulations >= 100
        
        sim_high = MonteCarloSimulator(n_simulations=20000)
        assert sim_high._n_simulations <= 10000

    def test_simulate_text_high_urgency(self) -> None:
        """Should simulate text input with high urgency."""
        sim = MonteCarloSimulator(n_simulations=1000)
        
        scores = {"urgency": 0.85, "sentiment": -0.6, "confidence": 0.8}
        
        result = sim.simulate(scores, InputType.TEXT, "infrastructure")
        
        assert result.n_simulations == 1000
        assert "critical" in result.severity_distribution or "warning" in result.severity_distribution
        assert result.expected_severity in ["critical", "warning", "info", "minor", "high"]
        assert 0.0 <= result.confidence_score <= 1.0

    def test_simulate_numeric_anomaly(self) -> None:
        """Should simulate numeric input with anomaly."""
        sim = MonteCarloSimulator(n_simulations=1000)
        
        scores = {"mean": 0.75, "confidence": 0.9, "structural_score": 0.7}
        
        result = sim.simulate(scores, InputType.NUMERIC, "infrastructure")
        
        assert result.n_simulations == 1000
        assert len(result.severity_distribution) > 0

    def test_severity_distribution_sums_to_one(self) -> None:
        """Severity distribution should sum to approximately 1.0."""
        sim = MonteCarloSimulator(n_simulations=1000)
        
        scores = {"urgency": 0.5, "confidence": 0.7}
        result = sim.simulate(scores, InputType.TEXT, "security")
        
        total_prob = sum(result.severity_distribution.values())
        assert 0.99 <= total_prob <= 1.01

    def test_confidence_interval_valid_bounds(self) -> None:
        """Confidence interval should have lower < upper."""
        sim = MonteCarloSimulator(n_simulations=1000)
        
        scores = {"urgency": 0.6}
        result = sim.simulate(scores, InputType.TEXT, "trading")
        
        lower, upper = result.confidence_interval
        assert lower <= upper
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0

    def test_n_simulations_produces_stable_distribution(self) -> None:
        """1000 simulations should produce stable distribution."""
        sim = MonteCarloSimulator(n_simulations=1000)
        
        scores = {"urgency": 0.7}
        
        result1 = sim.simulate(scores, InputType.TEXT, "infrastructure")
        result2 = sim.simulate(scores, InputType.TEXT, "infrastructure")
        
        # Distributions should be similar (within 10%)
        for severity in result1.severity_distribution:
            if severity in result2.severity_distribution:
                diff = abs(result1.severity_distribution[severity] - 
                          result2.severity_distribution[severity])
                assert diff < 0.1

    def test_uncertainty_level_classification(self) -> None:
        """Should classify uncertainty level correctly."""
        sim = MonteCarloSimulator(n_simulations=1000)
        
        # High variance scores → high uncertainty
        scores_high_var = {"urgency": 0.5, "confidence": 0.5}
        result_high = sim.simulate(scores_high_var, InputType.TEXT, "test")
        
        assert result_high.uncertainty_level in ["low", "moderate", "high"]

    def test_scenario_generation_best_worst_likely(self) -> None:
        """Should generate best/worst/most_likely scenarios."""
        sim = MonteCarloSimulator(n_simulations=1000)
        
        scores = {"urgency": 0.6}
        result = sim.simulate(scores, InputType.TEXT, "infrastructure")
        
        assert "best_case" in result.scenario_outcomes
        assert "worst_case" in result.scenario_outcomes
        assert "most_likely" in result.scenario_outcomes
        
        best = result.scenario_outcomes["best_case"]
        worst = result.scenario_outcomes["worst_case"]
        
        assert best["severity_score"] <= worst["severity_score"]

    def test_graceful_fail_invalid_scores(self) -> None:
        """Should raise ValueError for invalid scores."""
        sim = MonteCarloSimulator(n_simulations=1000)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            sim.simulate({}, InputType.TEXT, "test")

    def test_noise_model_text_vs_numeric(self) -> None:
        """TEXT should have higher noise than NUMERIC."""
        sim = MonteCarloSimulator(n_simulations=1000)
        
        sigma_text = sim._get_noise_sigma(InputType.TEXT)
        sigma_numeric = sim._get_noise_sigma(InputType.NUMERIC)
        
        assert sigma_text > sigma_numeric
        assert sigma_text == 0.10
        assert sigma_numeric == 0.05

    def test_performance_under_500ms(self) -> None:
        """1000 simulations should complete in < 500ms."""
        sim = MonteCarloSimulator(n_simulations=1000)
        
        scores = {"urgency": 0.7, "confidence": 0.8}
        
        start = time.monotonic()
        result = sim.simulate(scores, InputType.TEXT, "infrastructure")
        elapsed_ms = (time.monotonic() - start) * 1000
        
        assert elapsed_ms < 500
        assert result.n_simulations == 1000

    def test_edge_case_single_score(self) -> None:
        """Should handle single score input."""
        sim = MonteCarloSimulator(n_simulations=500)
        
        scores = {"urgency": 0.9}
        result = sim.simulate(scores, InputType.TEXT, "test")
        
        assert result.n_simulations == 500
        assert len(result.severity_distribution) > 0

    def test_mixed_input_type_noise(self) -> None:
        """MIXED type should use average noise."""
        sim = MonteCarloSimulator(n_simulations=1000)
        
        sigma_mixed = sim._get_noise_sigma(InputType.MIXED)
        assert sigma_mixed == 0.075

    def test_serialization_to_dict_complete(self) -> None:
        """to_dict should include all fields."""
        sim = MonteCarloSimulator(n_simulations=100)
        
        scores = {"urgency": 0.5}
        result = sim.simulate(scores, InputType.TEXT, "test")
        
        result_dict = result.to_dict()
        
        required_fields = [
            "n_simulations", "severity_distribution", "confidence_interval",
            "expected_severity", "confidence_score", "scenario_outcomes",
            "uncertainty_level"
        ]
        
        for field in required_fields:
            assert field in result_dict
