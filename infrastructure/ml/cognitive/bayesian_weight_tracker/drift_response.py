"""Gradual drift response for BayesianWeightTracker.

Consolidates: drift_response.py, drift_strategies.py.
"""

from __future__ import annotations

from typing import Dict, Optional

from .updater import GaussianPrior


_MILD_DRIFT_THRESHOLD: float = 0.7
_SEVERE_DRIFT_THRESHOLD: float = 1.5


def _apply_gradual_decay(
    accuracies: Dict[str, float],
    priors: Dict[str, GaussianPrior],
    decay_factor: float,
    variance_expansion: float,
) -> tuple[Dict[str, float], Dict[str, GaussianPrior]]:
    decayed = {k: v * decay_factor for k, v in accuracies.items()}
    expanded = {
        k: GaussianPrior(mu_0=v.mu_0 * decay_factor, sigma2_0=v.sigma2_0 * variance_expansion)
        for k, v in priors.items()
    }
    return decayed, expanded


def _apply_mild_drift_response(
    accuracies: Dict[str, float],
    priors: Dict[str, GaussianPrior],
) -> tuple[Dict[str, float], Dict[str, GaussianPrior]]:
    decayed = {k: v * 0.8 for k, v in accuracies.items()}
    expanded = {
        k: GaussianPrior(mu_0=v.mu_0 * 0.8, sigma2_0=v.sigma2_0 * 1.5)
        for k, v in priors.items()
    }
    return decayed, expanded


def _apply_severe_drift_response(
    accuracies: Dict[str, float],
    priors: Dict[str, GaussianPrior],
    decay_factor: float,
    variance_expansion: float,
) -> tuple[Dict[str, float], Dict[str, GaussianPrior]]:
    decayed = {k: v * decay_factor for k, v in accuracies.items()}
    expanded = {
        k: GaussianPrior(
            mu_0=0.5 * v.mu_0 * decay_factor + 0.5 * 0.5,
            sigma2_0=v.sigma2_0 * variance_expansion * 3.0,
        )
        for k, v in priors.items()
    }
    return decayed, expanded


class GradualDriftResponse:
    """Conditional severity-based drift response."""

    def __init__(
        self,
        decay_factor: float = 0.5,
        variance_expansion: float = 2.0,
    ) -> None:
        if not 0.0 < decay_factor < 1.0:
            raise ValueError(f"decay_factor must be in (0,1), got {decay_factor}")
        if variance_expansion < 1.0:
            raise ValueError(
                f"variance_expansion must be >= 1.0, got {variance_expansion}"
            )
        self._decay = decay_factor
        self._expansion = variance_expansion

    def apply_decay(
        self,
        accuracies: Dict[str, float],
        priors: Dict[str, GaussianPrior],
        drift_severity: Optional[float] = None,
    ) -> tuple[Dict[str, float], Dict[str, GaussianPrior]]:
        if drift_severity is None:
            return _apply_gradual_decay(
                accuracies, priors, self._decay, self._expansion,
            )
        if drift_severity >= _SEVERE_DRIFT_THRESHOLD:
            return _apply_severe_drift_response(
                accuracies, priors, self._decay, self._expansion,
            )
        return _apply_mild_drift_response(accuracies, priors)

    def should_apply_decay(
        self, drift_magnitude: float, threshold: float = 0.7,
    ) -> bool:
        return drift_magnitude >= threshold
