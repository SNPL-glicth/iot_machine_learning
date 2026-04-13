"""Tests for domain namespace isolation in BayesianWeightTracker."""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker import BayesianWeightTracker


class TestDomainNamespace:
    """Test suite for domain namespace isolation."""
    
    def test_iot_and_zenin_weights_isolated(self):
        """Test that IoT and Zenin weights are isolated."""
        # Create two trackers with different namespaces
        iot_tracker = BayesianWeightTracker(domain_namespace="iot")
        zenin_tracker = BayesianWeightTracker(domain_namespace="zenin")
        
        # Update IoT tracker
        iot_tracker.update(regime="stable", engine_name="taylor", prediction_error=0.1)
        
        # Update Zenin tracker with same regime but different error
        zenin_tracker.update(regime="stable", engine_name="universal", prediction_error=0.5)
        
        # Get weights from both
        iot_weights = iot_tracker.get_weights("stable", ["taylor", "universal"])
        zenin_weights = zenin_tracker.get_weights("stable", ["taylor", "universal"])
        
        # IoT should have history for taylor
        assert iot_tracker.has_history("stable")
        
        # Zenin should have history for universal
        assert zenin_tracker.has_history("stable")
        
        # Weights should be different (isolated)
        # IoT has only taylor data, Zenin has only universal data
        assert iot_weights != zenin_weights
    
    def test_default_namespace(self):
        """Test that default namespace works."""
        tracker = BayesianWeightTracker()  # No namespace specified
        
        tracker.update(regime="volatile", engine_name="ensemble", prediction_error=0.2)
        
        assert tracker.has_history("volatile")
        weights = tracker.get_weights("volatile", ["ensemble"])
        assert "ensemble" in weights
    
    def test_namespace_in_keys(self):
        """Test that namespace is actually used in internal keys."""
        tracker = BayesianWeightTracker(domain_namespace="test_domain")
        
        tracker.update(regime="noisy", engine_name="kalman", prediction_error=0.3)
        
        # Internal keys should have namespace prefix
        # Check that namespaced key exists in accuracy dict
        assert any("test_domain:" in key for key in tracker._accuracy.keys())
