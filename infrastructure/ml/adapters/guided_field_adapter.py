"""Ensemble Guided Field Adapter for Rosa Roja Engine.

Connects the existing FFT-based SeasonalPredictorEngine (Fourier macro gradient)
and NaiveBayesClassifier (historical prior densities) to the domain-level
GuidedFieldPort without violating hexagonal architectural boundaries.
"""

from __future__ import annotations

from collections import deque
import logging
import math
from typing import Dict, Optional, Tuple
import numpy as np

from domain.ports.rosa_roja.guided_field import GuidedFieldPort
from iot_machine_learning.infrastructure.ml.engines.seasonal.engine import SeasonalPredictorEngine
from iot_machine_learning.infrastructure.ml.inference.bayesian.naive_bayes import NaiveBayesClassifier

logger = logging.getLogger(__name__)


class EnsembleGuidedFieldAdapter(GuidedFieldPort):
    """Infrastructure adapter connecting existing Fourier and Bayesian engines to GuidedFieldPort."""

    def __init__(
        self,
        seasonal_engine: Optional[SeasonalPredictorEngine] = None,
        bayesian_classifier: Optional[NaiveBayesClassifier] = None,
        *,
        window_size: int = 50,
        default_dimension: int = 1,
    ) -> None:
        """Initialize adapter with existing analytical engines and observation buffers.

        Args:
            seasonal_engine: SeasonalPredictorEngine instance (Fourier FFT).
            bayesian_classifier: NaiveBayesClassifier instance (Bayesian priors).
            window_size: History window size for Fourier spectrum estimation.
            default_dimension: Default spatial dimensionality of state vector.
        """
        self._seasonal = seasonal_engine or SeasonalPredictorEngine()
        self._bayes = bayesian_classifier
        self._window_size = window_size
        self._default_dimension = default_dimension
        self._values: deque[float] = deque(maxlen=window_size)
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def record_observation(self, value: float, timestamp: Optional[float] = None) -> None:
        """Feed a new scalar observation and timestamp into the seasonal history window."""
        if math.isfinite(value):
            self._values.append(float(value))
            t = timestamp if (timestamp is not None and math.isfinite(timestamp)) else float(len(self._values))
            self._timestamps.append(t)

    def get_macro_gradient(self, timestamp: float) -> Tuple[np.ndarray, float]:
        """Calculates long-term macro trend vector and harmonic confidence from Fourier FFT.

        Args:
            timestamp: Epoch timestamp of current observation event.

        Returns:
            Tuple of:
                - np.ndarray: Normalized directional velocity vector in phase space.
                - float: Harmonic model confidence score in [0.0, 1.0].
        """
        min_required = self._seasonal._config.min_period * 2
        if len(self._values) < min_required:
            return np.zeros(self._default_dimension, dtype=np.float64), 0.0

        try:
            res = self._seasonal.predict(list(self._values), list(self._timestamps))
            if res is None or res.confidence < self._seasonal._config.min_confidence:
                return np.zeros(self._default_dimension, dtype=np.float64), 0.0

            last_val = self._values[-1] if self._values else 0.0
            direction = 0.0
            if res.predicted_value is not None:
                diff = res.predicted_value - last_val
                if abs(diff) > 1e-6:
                    direction = math.copysign(1.0, diff)

            if res.trend == "up" and direction <= 0.0:
                direction = 1.0
            elif res.trend == "down" and direction >= 0.0:
                direction = -1.0

            vec = np.zeros(self._default_dimension, dtype=np.float64)
            vec[0] = direction
            return vec, float(max(0.0, min(1.0, res.confidence)))
        except Exception as exc:
            logger.debug("guided_field_macro_gradient_failed: %s", exc)
            return np.zeros(self._default_dimension, dtype=np.float64), 0.0

    def get_motif_prior(self, regime: str) -> Dict[int, float]:
        """Queries historical global prior probability distribution across topological motifs.

        Args:
            regime: Current detected macro regime name.

        Returns:
            Dict mapping motif_id to prior probability mass (summing to 1.0).
        """
        if self._bayes is not None and self._bayes.classes and self._bayes.n_total > 0:
            probs: Dict[int, float] = {}
            total = float(self._bayes.n_total)
            for c_name, count in self._bayes.class_counts.items():
                try:
                    m_id = int(c_name)
                    probs[m_id] = count / total
                except (ValueError, TypeError):
                    continue
            if probs:
                mass = sum(probs.values())
                return {k: v / mass for k, v in probs.items()}

        return {}
