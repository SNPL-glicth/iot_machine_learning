"""
Model training for operational regime classifier.
"""

import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains models for regime classification."""
    
    def __init__(self, algorithm: str, n_components: int):
        """
        Initialize model trainer.
        
        Args:
            algorithm: Algorithm to use ("kmeans", "gmm", "hdbscan", "hmm")
            n_components: Number of clusters/components
        """
        self._algorithm = algorithm
        self._n_components = n_components
    
    def train(self, X, timestamps):
        """Train model with features."""
        if self._algorithm == "kmeans":
            return self._train_kmeans(X)
        elif self._algorithm == "gmm":
            return self._train_gmm(X)
        elif self._algorithm == "hdbscan":
            return self._train_hdbscan(X)
        elif self._algorithm == "hmm":
            return self._train_hmm(X, timestamps)
        else:
            raise ValueError(f"Unknown algorithm: {self._algorithm}")
    
    def _train_kmeans(self, X):
        """Train K-Means classifier."""
        try:
            from sklearn.cluster import KMeans
            
            model = KMeans(
                n_clusters=self._n_components,
                random_state=42,
                n_init=10,
            )
            model.fit(X)
            return model
            
        except ImportError:
            logger.warning("sklearn not available for K-Means")
            raise
    
    def _train_gmm(self, X):
        """Train GMM classifier."""
        try:
            from sklearn.mixture import GaussianMixture
            
            model = GaussianMixture(
                n_components=self._n_components,
                covariance_type='diag',
                random_state=42,
            )
            model.fit(X)
            return model
            
        except ImportError:
            logger.warning("sklearn not available for GMM")
            raise
    
    def _train_hdbscan(self, X):
        """Train HDBSCAN classifier."""
        try:
            import hdbscan
            
            model = hdbscan.HDBSCAN(
                min_cluster_size=50,
                min_samples=5,
            )
            model.fit(X)
            return model
            
        except ImportError:
            logger.warning("hdbscan not available")
            raise
    
    def _train_hmm(self, X, timestamps):
        """Train HMM classifier."""
        try:
            from hmmlearn import hmm
            
            model = hmm.GaussianHMM(
                n_components=self._n_components,
                covariance_type="diag",
                random_state=42,
            )
            model.fit(X)
            return model
            
        except ImportError:
            logger.warning("hmmlearn not available for HMM")
            raise
