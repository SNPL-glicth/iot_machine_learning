"""Error normalization for scale-invariant drift detection."""

from __future__ import annotations

from core.parameters.numerical_constants import EPSILON


class ErrorNormalizer:
    """Normalizes errors using z-score against rolling statistics."""
    
    def __init__(self, zscore_threshold: float = 3.0):
        self._zscore_threshold = zscore_threshold
    
    def normalize(
        self,
        error: float,
        rolling_mae: float,
        rolling_std: float,
        n_updates: int,
    ) -> float:
        """Normalize error to make drift detector scale-invariant.
        
        Uses z-score normalization against rolling statistics.
        With insufficient history (< 10 samples), returns raw error.
        """
        if n_updates < 10 or rolling_std < EPSILON.DIVISION:
            return error
        
        z = (error - rolling_mae) / rolling_std
        # Clamp extreme z-scores to prevent detector saturation
        return float(max(-10.0, min(10.0, z)))
    
    def compute_severity(self, normalized_error: float) -> float:
        """Compute drift severity from normalized error."""
        if self._zscore_threshold <= 0:
            return 0.0
        severity = abs(normalized_error) / self._zscore_threshold
        return min(1.0, max(0.0, severity))
    
    @property
    def zscore_threshold(self) -> float:
        return self._zscore_threshold