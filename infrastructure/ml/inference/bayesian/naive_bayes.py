"""Naive Bayes classifier for online multi-class classification.

P(class | features) ∝ P(features | class) * P(class)

Assumption: features are conditionally independent given class.
Online learning: incremental updates without full retraining.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from collections import defaultdict


@dataclass
class ClassProbabilities:
    """Class probability distribution."""
    probabilities: Dict[str, float]
    winner: str
    confidence: float
    
    def get_prob(self, class_name: str) -> float:
        """Get probability for class."""
        return self.probabilities.get(class_name, 0.0)


class NaiveBayesClassifier:
    """Online Naive Bayes classifier.
    
    Features are assumed Gaussian within each class.
    Works with 0 training examples (uniform prior).
    
    Example:
        clf = NaiveBayesClassifier(classes=["security", "infrastructure", "trading"])
        
        # Train online
        clf.fit_online({"urgency": 0.8, "sentiment": 0.2}, "security")
        clf.fit_online({"urgency": 0.3, "sentiment": 0.7}, "infrastructure")
        
        # Predict
        probs = clf.predict_proba({"urgency": 0.9, "sentiment": 0.1})
        # probs.winner = "security"
    """
    
    def __init__(self, classes: list[str] = None):
        """Initialize with optional class list.
        
        Args:
            classes: List of class names (optional, auto-discovered)
        """
        self.classes = set(classes) if classes else set()
        
        # P(class) — class priors
        self.class_counts: Dict[str, int] = defaultdict(int)
        
        # P(feature | class) — feature statistics per class
        # class → feature → {"sum": float, "sum_sq": float, "count": int}
        self.feature_stats: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0, "count": 0})
        )
        
        # Total training examples
        self.n_total = 0
    
    def fit_online(
        self,
        features: Dict[str, float],
        label: str,
    ) -> None:
        """Incremental update with one example.
        
        Args:
            features: Feature dict {name: value}
            label: Class label
        """
        # Add class if new
        self.classes.add(label)
        
        # Update class count
        self.class_counts[label] += 1
        self.n_total += 1
        
        # Update feature statistics for this class
        for feature_name, value in features.items():
            stats = self.feature_stats[label][feature_name]
            stats["sum"] += value
            stats["sum_sq"] += value ** 2
            stats["count"] += 1
    
    def predict_proba(
        self,
        features: Dict[str, float],
    ) -> ClassProbabilities:
        """Predict class probabilities.
        
        P(class | features) ∝ P(class) * Π P(feature_i | class)
        
        Args:
            features: Feature dict {name: value}
            
        Returns:
            ClassProbabilities with distribution
        """
        if not self.classes:
            # No training data → uniform prior
            return ClassProbabilities(
                probabilities={},
                winner="unknown",
                confidence=0.0,
            )
        
        log_probs = {}
        
        for class_name in self.classes:
            # Log P(class)
            class_count = self.class_counts.get(class_name, 0)
            
            if self.n_total > 0:
                # Laplace smoothing: (count + 1) / (total + num_classes)
                log_prior = np.log(
                    (class_count + 1) / (self.n_total + len(self.classes))
                )
            else:
                # Uniform prior
                log_prior = np.log(1.0 / len(self.classes))
            
            # Log P(features | class)
            log_likelihood = 0.0
            
            for feature_name, value in features.items():
                # Get feature statistics for this class
                stats = self.feature_stats[class_name].get(feature_name)
                
                if stats and stats["count"] > 0:
                    # Compute Gaussian likelihood
                    mean = stats["sum"] / stats["count"]
                    
                    # Variance with Bessel correction
                    if stats["count"] > 1:
                        variance = (
                            stats["sum_sq"] / stats["count"] - mean ** 2
                        )
                        variance = max(variance, 1e-4)  # Floor variance
                    else:
                        variance = 1.0  # Default variance
                    
                    # Log Gaussian likelihood
                    log_likelihood += -0.5 * (
                        ((value - mean) ** 2) / variance +
                        np.log(2 * np.pi * variance)
                    )
                else:
                    # No data for this feature-class pair
                    # Use neutral likelihood
                    log_likelihood += -0.5 * np.log(2 * np.pi)
            
            log_probs[class_name] = log_prior + log_likelihood
        
        # Convert log probabilities to probabilities
        max_log_prob = max(log_probs.values())
        
        # Subtract max for numerical stability
        exp_probs = {
            c: np.exp(lp - max_log_prob)
            for c, lp in log_probs.items()
        }
        
        # Normalize
        total = sum(exp_probs.values())
        probabilities = {
            c: p / total for c, p in exp_probs.items()
        }
        
        # Find winner
        winner = max(probabilities, key=probabilities.get)
        confidence = probabilities[winner]
        
        return ClassProbabilities(
            probabilities=probabilities,
            winner=winner,
            confidence=confidence,
        )
    
    def get_class_prior(self, class_name: str) -> float:
        """Get prior probability of class.
        
        Args:
            class_name: Class label
            
        Returns:
            P(class)
        """
        if self.n_total == 0:
            return 1.0 / max(len(self.classes), 1)
        
        count = self.class_counts.get(class_name, 0)
        return (count + 1) / (self.n_total + len(self.classes))
