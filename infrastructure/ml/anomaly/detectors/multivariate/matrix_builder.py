"""Matrix builder — construct multivariate matrices from time series.

Single responsibility: build and sanitize multivariate matrices for PCA.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MatrixBuilder:
    """Builds multivariate matrices from correlated time series.
    
    Responsibilities:
    - Determine common window size
    - Stack series into matrix columns
    - Sanitize NaN/Inf values
    """
    
    @staticmethod
    def build(
        target_values: List[float],
        correlated_series_data: Dict[str, List[float]],
        min_samples: int = 2,
    ) -> Optional[np.ndarray]:
        """Build multivariate matrix from target and correlated series.
        
        Args:
            target_values: Target series values.
            correlated_series_data: Dict of {series_id: values}.
            min_samples: Minimum samples required.
        
        Returns:
            Matrix X of shape (window_size, n_series) or None if invalid.
        """
        try:
            # Determine common window size (minimum length)
            window_size = len(target_values)
            for series_values in correlated_series_data.values():
                window_size = min(window_size, len(series_values))
            
            if window_size < min_samples:
                return None
            
            # Build matrix: columns = [target, correlated_1, correlated_2, ...]
            columns = [target_values[-window_size:]]
            
            for sid in sorted(correlated_series_data.keys()):
                series_values = correlated_series_data[sid]
                columns.append(series_values[-window_size:])
            
            # Stack columns into matrix
            X = np.column_stack(columns)
            
            # Sanitize: replace NaN/Inf with column mean
            X = MatrixBuilder._sanitize(X)
            
            return X
        
        except Exception as e:
            logger.error(
                "matrix_build_failed",
                extra={
                    "event": "MATRIX_BUILD_ERROR",
                    "error": str(e),
                },
            )
            return None
    
    @staticmethod
    def _sanitize(X: np.ndarray) -> np.ndarray:
        """Sanitize matrix by replacing NaN/Inf with column mean.
        
        Args:
            X: Input matrix.
        
        Returns:
            Sanitized matrix.
        """
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            mask = np.isfinite(col)
            if not mask.all():
                col_mean = np.mean(col[mask]) if mask.any() else 0.0
                X[~mask, col_idx] = col_mean
        
        return X
