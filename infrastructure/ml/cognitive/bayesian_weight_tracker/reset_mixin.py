"""Reset mixin for BayesianWeightTracker."""
from __future__ import annotations
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ResetMixin:
    """Mixin providing regime reset functionality."""

    def reset(self, regime: Optional[str] = None) -> None:
        """Clear accumulated accuracy data."""
        if regime is not None:
            self._accuracy.pop(regime, None)
        else:
            self._accuracy.clear()

    def reset_regime(
        self,
        regime: str,
        series_id: Optional[str] = None,
        use_gradual_decay: bool = True,
        drift_severity: Optional[float] = None,
    ) -> None:
        """Reset weights for a specific regime due to concept drift."""
        namespaced_regime = f"{self._domain_namespace}:{regime}"
        had_data = namespaced_regime in self._accuracy
        n_engines = len(self._accuracy.get(namespaced_regime, {}))
        if use_gradual_decay and had_data:
            decayed_acc, expanded_priors = self._drift_response.apply_decay(
                self._accuracy[namespaced_regime],
                self._priors[namespaced_regime],
                drift_severity=drift_severity,
            )
            self._accuracy[namespaced_regime] = decayed_acc
            self._priors[namespaced_regime] = expanded_priors
            logger.warning(
                "bayesian_weights_gradual_decay",
                extra={
                    "action": "gradual_decay",
                    "reason": "drift_detected",
                    "regime": regime,
                    "series_id": series_id or "unknown",
                    "n_engines": n_engines,
                    "decay_factor": self._drift_response._decay,
                    "timestamp": time.time(),
                },
            )
        else:
            self._accuracy.pop(namespaced_regime, None)
            self._priors.pop(namespaced_regime, None)
            self._regime_last_access.pop(namespaced_regime, None)
            self._regime_last_update.pop(namespaced_regime, None)
            logger.info(
                "bayesian_weights_regime_reset",
                extra={
                    "action": "regime_reset",
                    "reason": "drift_detected",
                    "regime": regime,
                    "series_id": series_id or "unknown",
                    "had_data": had_data,
                    "n_engines_cleared": n_engines,
                    "timestamp": time.time(),
                },
            )
