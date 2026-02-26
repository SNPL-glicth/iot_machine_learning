"""Interaction Field Service with Laplacian smoothing.

Models series interactions as a field with graph Laplacian for global consistency.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class InteractionFieldService:
    """Model series interactions as a field with Laplacian smoothing.
    
    Uses graph Laplacian to enforce consistency across correlated series.
    Laplacian: L = D - A where D is degree matrix, A is adjacency (correlation) matrix.
    """
    
    def __init__(self, correlation_matrix: Optional[object] = None, series_ids: Optional[List[str]] = None):
        """Initialize interaction field service.
        
        Args:
            correlation_matrix: Correlation matrix (numpy array or None)
            series_ids: List of series identifiers
        """
        if not HAS_NUMPY:
            self._available = False
            self._corr_matrix = None
            self._series_ids = []
            self._laplacian = None
            return
        
        self._available = True
        self._corr_matrix = correlation_matrix if correlation_matrix is not None else np.array([])
        self._series_ids = series_ids if series_ids is not None else []
        self._laplacian = None
        
        if self._corr_matrix is not None and len(self._corr_matrix) > 0:
            try:
                self._laplacian = self._compute_laplacian()
            except Exception:
                self._available = False
    
    def _compute_laplacian(self) -> object:
        """Compute graph Laplacian: L = D - A.
        
        D is degree matrix (diagonal with row sums of |A|)
        A is adjacency (correlation) matrix
        
        Returns:
            Laplacian matrix (numpy array)
        """
        if not HAS_NUMPY or self._corr_matrix is None:
            return None
        
        degree = np.diag(np.sum(np.abs(self._corr_matrix), axis=1))
        return degree - self._corr_matrix
    
    def smooth_predictions(
        self,
        predictions: Dict[str, float],
        smoothing_factor: float = 0.3,
    ) -> Dict[str, float]:
        """Apply Laplacian smoothing to predictions.
        
        Formula: φ_smoothed = φ - α·L·φ
        where L is Laplacian, α controls smoothing strength.
        
        This enforces that correlated series have similar predictions.
        
        Args:
            predictions: Dict mapping series_id to prediction value
            smoothing_factor: Smoothing strength (0 = no smoothing, 1 = full smoothing)
        
        Returns:
            Smoothed predictions dict
        """
        if not self._available or self._laplacian is None:
            return predictions
        
        if not predictions:
            return predictions
        
        try:
            pred_vector = np.array([predictions.get(sid, 0.0) for sid in self._series_ids])
            
            if len(pred_vector) == 0:
                return predictions
            
            smoothed_vector = pred_vector - smoothing_factor * (self._laplacian @ pred_vector)
            
            return {
                sid: float(smoothed_vector[i])
                for i, sid in enumerate(self._series_ids)
                if sid in predictions
            }
        except Exception:
            return predictions
