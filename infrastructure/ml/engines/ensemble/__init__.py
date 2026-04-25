"""Ensemble predictor — weighted combination of multiple engines.

DEPRECATED: EnsembleWeightedPredictor moved to deprecated/.
Use WeightedFusion + InhibitionGate instead.
"""

from ..deprecated.ensemble_predictor import EnsembleWeightedPredictor

__all__ = ["EnsembleWeightedPredictor"]
