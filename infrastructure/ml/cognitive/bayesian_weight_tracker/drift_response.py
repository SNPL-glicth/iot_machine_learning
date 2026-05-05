"""Gradual Drift Response — conditional severity-based response.

Replaces blind amnesia with a statistically principled approach:
- Mild drift: gradual decay to preserve learned structure
- Severe drift: partial reset blending old priors with defaults

Preserves prior structure using Bayesian blending instead of
simple multiplicative decay.
"""

from __future__ import annotations

from typing import Dict, Optional

from iot_machine_learning.infrastructure.ml.inference.bayesian.prior import GaussianPrior

from .drift_strategies import (
    _apply_gradual_decay,
    _apply_mild_drift_response,
    _apply_severe_drift_response,
)


# Severity thresholds for conditional response
_MILD_DRIFT_THRESHOLD: float = 0.7
_SEVERE_DRIFT_THRESHOLD: float = 1.5


class GradualDriftResponse:
    """Handles concept drift via conditional severity-based response.

    Attributes:
        decay_factor: Multiplier for accuracy (0.5 = halve weights)
        variance_expansion: Multiplier for σ² (2.0 = double uncertainty)
    """

    def __init__(
        self,
        decay_factor: float = 0.5,
        variance_expansion: float = 2.0,
    ) -> None:
        """Initialize drift response.

        Args:
            decay_factor: Weight decay ∈ (0, 1). 0.5 = halve, 0.8 = gentle
            variance_expansion: Uncertainty increase ≥ 1.0. 2.0 = double σ²
        """
        if not 0.0 < decay_factor < 1.0:
            raise ValueError(f"decay_factor must be in (0,1), got {decay_factor}")
        if variance_expansion < 1.0:
            raise ValueError(f"variance_expansion must be ≥1.0, got {variance_expansion}")

        self._decay = decay_factor
        self._expansion = variance_expansion

    def apply_decay(
        self,
        accuracies: Dict[str, float],
        priors: Dict[str, GaussianPrior],
        drift_severity: Optional[float] = None,
    ) -> tuple[Dict[str, float], Dict[str, GaussianPrior]]:
        """Apply conditional drift response based on severity.

        Statistical rationale:
        - No severity (None): backward-compatible gradual decay
        - Mild drift (severity < 1.5): gentle decay (factor 0.8), mild
          variance expansion (1.5×). Preserves most learned structure.
        - Severe drift (severity ≥ 1.5): partial reset via Bayesian blending
          of old priors with default priors (blend alpha = 0.5), strong
          variance expansion (3.0×) to rapidly admit new data.

        Args:
            accuracies: Current accuracy dict {engine: accuracy}
            priors: Current priors dict {engine: GaussianPrior}
            drift_severity: Normalized drift severity ≥ 0. None triggers
                backward-compatible default decay.

        Returns:
            (decayed_accuracies, expanded_priors)
        """
        if drift_severity is None:
            return _apply_gradual_decay(accuracies, priors, self._decay, self._expansion)
        if drift_severity >= _SEVERE_DRIFT_THRESHOLD:
            return _apply_severe_drift_response(accuracies, priors, self._decay, self._expansion)
        return _apply_mild_drift_response(accuracies, priors)

    def should_apply_decay(
        self,
        drift_magnitude: float,
        threshold: float = 0.7,
    ) -> bool:
        """Determine if decay should be applied.

        Args:
            drift_magnitude: Drift score from detector
            threshold: Minimum magnitude to trigger decay

        Returns:
            True if decay should be applied
        """
        return drift_magnitude >= threshold
