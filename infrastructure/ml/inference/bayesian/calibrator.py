"""Probability calibration via Platt scaling.

Transforms raw scores → calibrated probabilities.
P_calibrated = sigmoid(a * score + b)

Fixes overconfident predictions from heuristic analyzers.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import minimize


@dataclass
class CalibratedScores:
    """Calibration result."""
    calibrated: np.ndarray
    raw: np.ndarray
    a: float  # Platt scaling parameter
    b: float  # Platt scaling parameter
    
    def get_calibrated(self, index: int) -> float:
        """Get calibrated score at index."""
        return float(self.calibrated[index])


class ProbabilityCalibrator:
    """Platt scaling for probability calibration.
    
    Learns sigmoid transformation: P = 1 / (1 + exp(-(a*score + b)))
    
    Example:
        calibrator = ProbabilityCalibrator()
        
        # Train
        raw_scores = np.array([0.2, 0.8, 0.5, 0.9])
        true_labels = np.array([0, 1, 0, 1])
        calibrator.calibrate(raw_scores, true_labels)
        
        # Transform new score
        calibrated = calibrator.transform(0.7)
    """
    
    def __init__(self):
        self.a: Optional[float] = None
        self.b: Optional[float] = None
        self._is_fitted = False
    
    def calibrate(
        self,
        raw_scores: np.ndarray,
        true_labels: np.ndarray,
    ) -> CalibratedScores:
        """Fit Platt scaling parameters via MLE.
        
        Minimizes negative log-likelihood:
        -Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
        where p_i = sigmoid(a * score_i + b)
        
        Args:
            raw_scores: Raw scores (any range)
            true_labels: Binary labels (0 or 1)
            
        Returns:
            CalibratedScores with fitted parameters
        """
        raw_scores = np.asarray(raw_scores).flatten()
        true_labels = np.asarray(true_labels).flatten()
        
        if len(raw_scores) != len(true_labels):
            raise ValueError("Scores and labels must have same length")
        
        if len(raw_scores) < 2:
            # Not enough data → identity transform
            self.a = 1.0
            self.b = 0.0
            self._is_fitted = True
            return CalibratedScores(
                calibrated=raw_scores,
                raw=raw_scores,
                a=1.0,
                b=0.0,
            )
        
        # Define negative log-likelihood
        def neg_log_likelihood(params):
            a, b = params
            logits = a * raw_scores + b
            
            # Sigmoid with numerical stability
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
            
            # Clip probabilities to avoid log(0)
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            
            # Binary cross-entropy
            loss = -np.sum(
                true_labels * np.log(probs) +
                (1 - true_labels) * np.log(1 - probs)
            )
            
            return loss
        
        # Optimize
        initial_params = [1.0, 0.0]
        result = minimize(
            neg_log_likelihood,
            initial_params,
            method='BFGS',
        )
        
        self.a, self.b = result.x
        self._is_fitted = True
        
        # Compute calibrated scores
        calibrated = self.transform(raw_scores)
        
        return CalibratedScores(
            calibrated=calibrated,
            raw=raw_scores,
            a=self.a,
            b=self.b,
        )
    
    def transform(self, raw_score: np.ndarray | float) -> np.ndarray | float:
        """Apply calibration to raw score(s).
        
        Args:
            raw_score: Raw score or array of scores
            
        Returns:
            Calibrated probability in [0, 1]
        """
        if not self._is_fitted:
            # Not fitted → return identity
            return raw_score
        
        is_scalar = np.isscalar(raw_score)
        scores = np.atleast_1d(raw_score)
        
        # Apply sigmoid transformation
        logits = self.a * scores + self.b
        calibrated = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        
        if is_scalar:
            return float(calibrated[0])
        
        return calibrated
    
    def is_fitted(self) -> bool:
        """Check if calibrator has been fitted."""
        return self._is_fitted
