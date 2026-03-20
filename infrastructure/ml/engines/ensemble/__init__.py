"""Ensemble predictor — weighted combination of multiple engines.

Components:
    - EnsembleWeightedPredictor: Combines multiple PredictionPort engines with dynamic weights
"""

from .predictor import EnsembleWeightedPredictor

__all__ = ["EnsembleWeightedPredictor"]
