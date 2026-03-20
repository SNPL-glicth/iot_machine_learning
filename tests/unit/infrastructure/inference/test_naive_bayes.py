"""Tests for Naive Bayes classifier."""

import pytest
import numpy as np

from iot_machine_learning.infrastructure.ml.inference.bayesian import (
    NaiveBayesClassifier,
    ClassProbabilities,
)


class TestNaiveBayesBasic:
    """Test basic Naive Bayes functionality."""
    
    def test_empty_classifier(self):
        """Test classifier with no training data."""
        clf = NaiveBayesClassifier()
        
        features = {"urgency": 0.8, "sentiment": 0.3}
        probs = clf.predict_proba(features)
        
        assert probs.winner == "unknown"
        assert probs.confidence == 0.0
    
    def test_single_class_training(self):
        """Test training with single class."""
        clf = NaiveBayesClassifier()
        
        clf.fit_online({"urgency": 0.8, "sentiment": 0.2}, "security")
        clf.fit_online({"urgency": 0.9, "sentiment": 0.1}, "security")
        
        probs = clf.predict_proba({"urgency": 0.85, "sentiment": 0.15})
        
        assert probs.winner == "security"
        assert probs.confidence > 0.5
    
    def test_multi_class_training(self):
        """Test training with multiple classes."""
        clf = NaiveBayesClassifier()
        
        # Train security examples (high urgency, low sentiment)
        for _ in range(20):
            clf.fit_online({"urgency": 0.9, "sentiment": 0.1}, "security")
        
        # Train infrastructure examples (low urgency, high sentiment)
        for _ in range(20):
            clf.fit_online({"urgency": 0.3, "sentiment": 0.7}, "infrastructure")
        
        # Verify both classes were learned
        assert "security" in clf.classes
        assert "infrastructure" in clf.classes
        assert len(clf.classes) == 2
        
        # Predictions should return valid probabilities
        probs = clf.predict_proba({"urgency": 0.95, "sentiment": 0.05})
        assert probs.winner in ["security", "infrastructure"]
        assert 0.0 <= probs.confidence <= 1.0


class TestNaiveBayesOnlineLearning:
    """Test online learning capabilities."""
    
    def test_incremental_updates(self):
        """Test incremental updates without retraining."""
        clf = NaiveBayesClassifier()
        
        # Initial training
        clf.fit_online({"x": 1.0}, "A")
        assert clf.n_total == 1
        
        # Add more examples
        clf.fit_online({"x": 1.5}, "A")
        assert clf.n_total == 2
        
        clf.fit_online({"x": -1.0}, "B")
        assert clf.n_total == 3
        
        # Verify class counts
        assert clf.class_counts["A"] == 2
        assert clf.class_counts["B"] == 1
    
    def test_new_class_discovery(self):
        """Test auto-discovery of new classes."""
        clf = NaiveBayesClassifier()
        
        clf.fit_online({"x": 1.0}, "A")
        assert "A" in clf.classes
        
        clf.fit_online({"x": -1.0}, "B")
        assert "B" in clf.classes
        
        assert len(clf.classes) == 2


class TestNaiveBayesProbabilities:
    """Test probability computation."""
    
    def test_probability_distribution(self):
        """Test probabilities sum to 1."""
        clf = NaiveBayesClassifier()
        
        clf.fit_online({"x": 1.0}, "A")
        clf.fit_online({"x": -1.0}, "B")
        
        probs = clf.predict_proba({"x": 0.5})
        
        # Probabilities should sum to ~1.0
        total = sum(probs.probabilities.values())
        assert abs(total - 1.0) < 1e-6
    
    def test_confidence_in_range(self):
        """Test confidence is in [0, 1]."""
        clf = NaiveBayesClassifier()
        
        for i in range(10):
            clf.fit_online({"x": float(i)}, "A")
        
        probs = clf.predict_proba({"x": 5.0})
        
        assert 0.0 <= probs.confidence <= 1.0
    
    def test_get_prob_method(self):
        """Test get_prob accessor."""
        clf = NaiveBayesClassifier()
        
        clf.fit_online({"x": 1.0}, "A")
        clf.fit_online({"x": -1.0}, "B")
        
        probs = clf.predict_proba({"x": 0.5})
        
        prob_a = probs.get_prob("A")
        assert 0.0 <= prob_a <= 1.0
        
        prob_missing = probs.get_prob("C")
        assert prob_missing == 0.0


class TestNaiveBayesFeatures:
    """Test feature handling."""
    
    def test_multiple_features(self):
        """Test with multiple features."""
        clf = NaiveBayesClassifier()
        
        # Train with 3 features
        clf.fit_online({"urgency": 0.9, "sentiment": 0.1, "length": 100}, "security")
        clf.fit_online({"urgency": 0.3, "sentiment": 0.7, "length": 50}, "operations")
        
        probs = clf.predict_proba({"urgency": 0.85, "sentiment": 0.15, "length": 90})
        
        assert probs.winner in ["security", "operations"]
    
    def test_missing_features(self):
        """Test prediction with missing features."""
        clf = NaiveBayesClassifier()
        
        # Train with feature 'x'
        clf.fit_online({"x": 1.0, "y": 2.0}, "A")
        
        # Predict with only 'x' (missing 'y')
        probs = clf.predict_proba({"x": 1.0})
        
        assert probs.winner == "A"
    
    def test_new_features_at_prediction(self):
        """Test prediction with previously unseen features."""
        clf = NaiveBayesClassifier()
        
        # Train with feature 'x'
        clf.fit_online({"x": 1.0}, "A")
        
        # Predict with new feature 'z'
        probs = clf.predict_proba({"x": 1.0, "z": 5.0})
        
        assert probs.winner == "A"


class TestNaiveBayesPriors:
    """Test prior probability computation."""
    
    def test_uniform_prior_no_data(self):
        """Test uniform prior with no training data."""
        clf = NaiveBayesClassifier(classes=["A", "B", "C"])
        
        prior_a = clf.get_class_prior("A")
        prior_b = clf.get_class_prior("B")
        
        # Should be uniform
        assert abs(prior_a - 1.0/3.0) < 1e-6
        assert abs(prior_b - 1.0/3.0) < 1e-6
    
    def test_prior_with_data(self):
        """Test prior updates with training data."""
        clf = NaiveBayesClassifier()
        
        # Train 3 examples of A, 1 of B
        for _ in range(3):
            clf.fit_online({"x": 1.0}, "A")
        
        clf.fit_online({"x": -1.0}, "B")
        
        prior_a = clf.get_class_prior("A")
        prior_b = clf.get_class_prior("B")
        
        # A should have higher prior
        assert prior_a > prior_b


class TestNaiveBayesEdgeCases:
    """Test edge cases."""
    
    def test_single_feature_single_value(self):
        """Test with constant feature values."""
        clf = NaiveBayesClassifier()
        
        # All examples have x=1.0
        for _ in range(5):
            clf.fit_online({"x": 1.0}, "A")
        
        probs = clf.predict_proba({"x": 1.0})
        
        assert probs.winner == "A"
    
    def test_zero_variance_feature(self):
        """Test with zero variance in feature."""
        clf = NaiveBayesClassifier()
        
        # All A examples have x=1.0
        clf.fit_online({"x": 1.0}, "A")
        clf.fit_online({"x": 1.0}, "A")
        
        # B example has different value
        clf.fit_online({"x": 2.0}, "B")
        
        probs = clf.predict_proba({"x": 1.0})
        
        # Should still work despite zero variance in A
        assert probs.winner in ["A", "B"]
