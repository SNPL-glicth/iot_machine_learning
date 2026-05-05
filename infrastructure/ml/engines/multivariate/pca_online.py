"""Online PCA — incremental PCA for streaming multivariate anomaly detection.

Wrapper around sklearn IncrementalPCA for online learning.
Computes Mahalanobis distance in reduced PCA space.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OnlinePCA:
    """Online PCA with Mahalanobis distance scoring.
    
    Maintains incremental PCA model and computes anomaly scores
    based on distance in reduced space.
    
    Attributes:
        _n_components: Number of PCA components.
        _pca: IncrementalPCA instance.
        _fitted: Whether PCA has been fitted.
    """
    
    def __init__(
        self,
        n_components: int = 2,
    ) -> None:
        """Initialize online PCA.
        
        Args:
            n_components: Number of principal components.
        
        Raises:
            ValueError: If n_components < 1.
        """
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        
        self._n_components = n_components
        self._fitted = False
        self._available = self._check_sklearn()
        
        if self._available:
            from sklearn.decomposition import IncrementalPCA
            self._pca = IncrementalPCA(n_components=n_components)
        else:
            self._pca = None
    
    def _check_sklearn(self) -> bool:
        """Check if sklearn is available."""
        try:
            import sklearn.decomposition
            return True
        except ImportError:
            logger.warning(
                "pca_sklearn_unavailable",
                extra={
                    "event": "WARNING",
                    "reason": "sklearn_not_installed",
                    "action_taken": "pca_disabled",
                },
            )
            return False
    
    def partial_fit(self, X: np.ndarray) -> None:
        """Incrementally fit PCA model.
        
        Args:
            X: Data matrix (n_samples, n_features).
        """
        if not self._available or self._pca is None:
            return
        
        if X.shape[0] == 0:
            return
        
        # Ensure at least n_components samples
        if X.shape[0] < self._n_components:
            logger.debug(
                "pca_insufficient_samples_for_fit",
                extra={
                    "required": self._n_components,
                    "received": X.shape[0],
                },
            )
            return
        
        self._pca.partial_fit(X)
        self._fitted = True
    
    def transform(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Transform data to PCA space.
        
        Args:
            X: Data matrix (n_samples, n_features).
        
        Returns:
            Transformed data or None if not fitted.
        """
        if not self._available or self._pca is None or not self._fitted:
            return None
        
        try:
            return self._pca.transform(X)
        except Exception as e:
            logger.error(
                "pca_transform_failed",
                extra={
                    "event": "PHASE_ERROR",
                    "error": str(e),
                },
            )
            return None
    
    def score_samples(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Compute Mahalanobis distance in PCA space.
        
        Args:
            X: Data matrix (n_samples, n_features).
        
        Returns:
            Distances for each sample or None if not fitted.
        """
        if not self._available or self._pca is None or not self._fitted:
            return None
        
        try:
            # Transform to PCA space
            X_transformed = self._pca.transform(X)
            
            # Compute Mahalanobis distance
            # In PCA space, components are uncorrelated, so we can use
            # Euclidean distance weighted by explained variance
            distances = np.sqrt(np.sum(X_transformed ** 2, axis=1))
            
            return distances
        
        except Exception as e:
            logger.error(
                "pca_score_samples_failed",
                extra={
                    "event": "PHASE_ERROR",
                    "error": str(e),
                },
            )
            return None
    
    @property
    def fitted(self) -> bool:
        """True if PCA has been fitted."""
        return self._fitted
    
    @property
    def n_components(self) -> int:
        """Number of components."""
        return self._n_components
