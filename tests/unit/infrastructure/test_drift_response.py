"""Tests for Gradual Drift Response."""

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.drift_response import (
    GradualDriftResponse,
)
from iot_machine_learning.infrastructure.ml.inference.bayesian.prior import GaussianPrior


class TestGradualDriftResponse:
    """Test gradual decay instead of full reset."""
    
    def test_initialization_validates_decay_factor(self):
        """Decay factor must be in (0, 1)."""
        with pytest.raises(ValueError, match="decay_factor must be in"):
            GradualDriftResponse(decay_factor=0.0)
        
        with pytest.raises(ValueError, match="decay_factor must be in"):
            GradualDriftResponse(decay_factor=1.5)
    
    def test_initialization_validates_variance_expansion(self):
        """Variance expansion must be >= 1.0."""
        with pytest.raises(ValueError, match="variance_expansion must be"):
            GradualDriftResponse(variance_expansion=0.5)
    
    def test_apply_decay_reduces_accuracies(self):
        """Decay should reduce all accuracies by decay_factor."""
        response = GradualDriftResponse(decay_factor=0.5, variance_expansion=2.0)
        
        accuracies = {"engine_a": 0.8, "engine_b": 0.6}
        priors = {
            "engine_a": GaussianPrior(mu_0=0.8, sigma2_0=1.0),
            "engine_b": GaussianPrior(mu_0=0.6, sigma2_0=1.0),
        }
        
        decayed_acc, expanded_priors = response.apply_decay(accuracies, priors)
        
        assert decayed_acc["engine_a"] == pytest.approx(0.4, abs=1e-6)
        assert decayed_acc["engine_b"] == pytest.approx(0.3, abs=1e-6)
    
    def test_apply_decay_expands_variance(self):
        """Decay should expand variance by expansion_factor."""
        response = GradualDriftResponse(decay_factor=0.5, variance_expansion=2.0)
        
        accuracies = {"engine_a": 0.8}
        priors = {"engine_a": GaussianPrior(mu_0=0.8, sigma2_0=1.0)}
        
        _, expanded_priors = response.apply_decay(accuracies, priors)
        
        # μ_0 = 0.8 * 0.5 = 0.4
        # σ²_0 = 1.0 * 2.0 = 2.0
        assert expanded_priors["engine_a"].get_param("mu_0") == pytest.approx(0.4, abs=1e-6)
        assert expanded_priors["engine_a"].get_param("sigma2_0") == pytest.approx(2.0, abs=1e-6)
    
    def test_apply_decay_preserves_keys(self):
        """Decay should preserve all engine keys."""
        response = GradualDriftResponse(decay_factor=0.8, variance_expansion=1.5)
        
        accuracies = {"e1": 0.9, "e2": 0.7, "e3": 0.5}
        priors = {
            "e1": GaussianPrior(mu_0=0.9, sigma2_0=1.0),
            "e2": GaussianPrior(mu_0=0.7, sigma2_0=1.0),
            "e3": GaussianPrior(mu_0=0.5, sigma2_0=1.0),
        }
        
        decayed_acc, expanded_priors = response.apply_decay(accuracies, priors)
        
        assert set(decayed_acc.keys()) == {"e1", "e2", "e3"}
        assert set(expanded_priors.keys()) == {"e1", "e2", "e3"}
    
    def test_should_apply_decay_threshold(self):
        """should_apply_decay checks drift magnitude."""
        response = GradualDriftResponse()
        
        assert response.should_apply_decay(drift_magnitude=0.8, threshold=0.7) is True
        assert response.should_apply_decay(drift_magnitude=0.6, threshold=0.7) is False
        assert response.should_apply_decay(drift_magnitude=0.7, threshold=0.7) is True
    
    def test_gentle_decay_preserves_more_information(self):
        """Gentle decay (0.8) should preserve more than aggressive (0.5)."""
        gentle = GradualDriftResponse(decay_factor=0.8)
        aggressive = GradualDriftResponse(decay_factor=0.5)
        
        accuracies = {"e1": 0.9}
        priors = {"e1": GaussianPrior(mu_0=0.9, sigma2_0=1.0)}
        
        gentle_acc, _ = gentle.apply_decay(accuracies, priors)
        aggressive_acc, _ = aggressive.apply_decay(accuracies, priors)
        
        assert gentle_acc["e1"] > aggressive_acc["e1"]
    
    def test_empty_dicts_handled(self):
        """Empty accuracies/priors should return empty."""
        response = GradualDriftResponse()
        
        decayed_acc, expanded_priors = response.apply_decay({}, {})
        
        assert decayed_acc == {}
        assert expanded_priors == {}
