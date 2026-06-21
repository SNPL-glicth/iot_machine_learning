"""
OperationalRegimeClassifier for classifying operational regimes.

Supports multiple algorithms (K-Means, GMM, HDBSCAN, HMM) with heuristic fallback.
"""

from typing import Optional, List, TYPE_CHECKING
import logging

from .models.regime_prediction import RegimePrediction
from .models.regime_config import RegimeConfig
from .training import ModelTrainer
from .prediction import ModelPredictor
from .heuristic import HeuristicClassifier

if TYPE_CHECKING:
    from infrastructure.ml.cognitive.dynamic.models.dynamic_features import DynamicFeatures

logger = logging.getLogger(__name__)


class OperationalRegimeClassifier:
    """Classifier for operational regimes with multiple algorithm support."""
    
    # Regime names mapping
    REGIME_NAMES = {
        0: "STABLE_LOW",
        1: "STABLE_NORMAL",
        2: "STABLE_HIGH",
        3: "VOLATILE_PEAK",
        4: "TRANSITIONAL",
    }
    
    def __init__(
        self,
        algorithm: str = "kmeans",  # "kmeans", "gmm", "hdbscan", "hmm"
        n_components: int = 5,
        smoothing_window: int = 5,
    ):
        """
        Initialize operational regime classifier.
        
        Args:
            algorithm: Algorithm to use ("kmeans", "gmm", "hdbscan", "hmm")
            n_components: Number of clusters/components
            smoothing_window: Smoothing window for derivatives
        """
        self._algorithm = algorithm
        self._n_components = n_components
        self._smoothing_window = smoothing_window
        self._model = None
        self._is_trained = False
        
        self._trainer = ModelTrainer(algorithm, n_components)
        self._predictor = ModelPredictor(algorithm, REGIME_NAMES)
        self._heuristic = HeuristicClassifier()
    
    def train(
        self,
        features_matrix: List[List[float]],
        timestamps: List[float],
    ) -> None:
        """Train classifier with historical features."""
        try:
            import numpy as np
            X = np.array(features_matrix)
            
            self._model = self._trainer.train(X, timestamps)
            self._is_trained = True
            logger.info(f"Regime classifier trained with {self._algorithm}")
            
        except ImportError as e:
            logger.warning(f"Failed to import required library: {e}")
            self._is_trained = False
        except Exception as e:
            logger.error(f"Failed to train classifier: {e}")
            self._is_trained = False
    
    def classify(
        self,
        dynamic_features: 'DynamicFeatures',
        config: RegimeConfig,
        current_value: float = None,
    ) -> RegimePrediction:
        """Classify DynamicFeatures into operational regime."""
        if not self._is_trained:
            return self._heuristic.classify(dynamic_features, config, current_value)
        
        feature_vector = self._extract_features(dynamic_features, config)
        if feature_vector is None:
            return self._heuristic.classify(dynamic_features, config, current_value)
        
        try:
            import numpy as np
            X = np.array([feature_vector])
            regime_idx, confidence = self._predictor.predict(X, self._model)
            regime_name = self._predictor.map_idx_to_regime(regime_idx)
            
            return RegimePrediction(
                regime=regime_name,
                confidence=confidence,
                cluster_idx=regime_idx,
            )
        except Exception as e:
            logger.warning(f"Failed to predict with model: {e}, using heuristic")
            return self._heuristic.classify(dynamic_features, config, current_value)
    
    def _extract_features(
        self,
        dynamic_features: 'DynamicFeatures',
        config: RegimeConfig,
    ) -> Optional[List[float]]:
        """Extract feature vector from DynamicFeatures."""
        features = []
        
        if config.use_derivative and dynamic_features.derivative is not None:
            features.append(dynamic_features.derivative)
        else:
            return None
        
        if config.use_rolling_std and dynamic_features.rolling_std_1h is not None:
            features.append(dynamic_features.rolling_std_1h)
        else:
            return None
        
        if config.use_second_derivative and dynamic_features.second_derivative is not None:
            features.append(dynamic_features.second_derivative)
        
        if config.use_lag_features and dynamic_features.lag_6 is not None:
            features.append(dynamic_features.lag_6)
        
        return features if features else None
    
    @property
    def is_trained(self) -> bool:
        """Check if classifier is trained."""
        return self._is_trained
