"""
Model prediction for operational regime classifier.
"""

import logging

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Predicts regimes using trained models."""
    
    def __init__(self, algorithm: str, regime_names: dict):
        """
        Initialize model predictor.
        
        Args:
            algorithm: Algorithm to use ("kmeans", "gmm", "hdbscan", "hmm")
            regime_names: Mapping from cluster index to regime name
        """
        self._algorithm = algorithm
        self._regime_names = regime_names
    
    def predict(self, X, model):
        """Predict regime using trained model."""
        if self._algorithm == "kmeans":
            return self._predict_kmeans(X, model)
        elif self._algorithm == "gmm":
            return self._predict_gmm(X, model)
        elif self._algorithm == "hdbscan":
            return self._predict_hdbscan(X, model)
        elif self._algorithm == "hmm":
            return self._predict_hmm(X, model)
        else:
            raise ValueError(f"Unknown algorithm: {self._algorithm}")
    
    def _predict_kmeans(self, X, model):
        """Predict with K-Means."""
        regime_idx = int(model.predict(X)[0])
        distances = model.transform(X)[0]
        min_dist = distances[regime_idx]
        confidence = 1.0 / (1.0 + min_dist)
        return regime_idx, confidence
    
    def _predict_gmm(self, X, model):
        """Predict with GMM."""
        regime_idx = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]
        confidence = probs[regime_idx]
        return regime_idx, confidence
    
    def _predict_hdbscan(self, X, model):
        """Predict with HDBSCAN."""
        labels = model.labels_
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(model._raw_data)
        distances, indices = nbrs.kneighbors(X)
        regime_idx = labels[indices[0][0]]
        
        if regime_idx == -1:
            return 4, 0.5  # TRANSITIONAL with low confidence
        
        confidence = 1.0 / (1.0 + distances[0][0])
        return regime_idx, confidence
    
    def _predict_hmm(self, X, model):
        """Predict with HMM."""
        regime_idx = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]
        confidence = probs[regime_idx]
        return regime_idx, confidence
    
    def map_idx_to_regime(self, idx: int) -> str:
        """Map cluster index to regime name."""
        return self._regime_names.get(idx, "UNKNOWN")
