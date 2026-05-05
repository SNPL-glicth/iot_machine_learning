"""Update mixin for BayesianWeightTracker."""
from __future__ import annotations
import logging
import math
import time
from typing import Optional
import numpy as np
from iot_machine_learning.infrastructure.ml.inference.bayesian.prior import GaussianPrior
from .constants import _PERSIST_EVERY_N_UPDATES, WeightTrackerConfig
from .regularization import apply_l2_regularization, compute_regularization_strength

logger = logging.getLogger(__name__)


class UpdateMixin:
    """Mixin providing the Bayesian update cycle."""

    def update(
        self,
        regime: str,
        engine_name: str,
        prediction_error: float,
        alpha: Optional[float] = None,
        *,
        series_id: Optional[str] = None,
        drift_score: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._update_inner(
                regime, engine_name, prediction_error, alpha,
                series_id=series_id, drift_score=drift_score,
            )

    def _update_inner(
        self,
        regime: str,
        engine_name: str,
        prediction_error: float,
        alpha: Optional[float] = None,
        *,
        series_id: Optional[str] = None,
        drift_score: Optional[float] = None,
    ) -> None:
        namespaced_regime = f"{self._domain_namespace}:{regime}"
        if namespaced_regime not in self._accuracy and len(self._accuracy) >= self._config.max_regimes:
            coldest = min(self._regime_last_access, key=self._regime_last_access.get)
            del self._accuracy[coldest]
            del self._regime_last_access[coldest]
            self._regime_last_update.pop(coldest, None)
        now = time.monotonic()
        self._regime_last_access[namespaced_regime] = now
        self._regime_last_update[namespaced_regime] = now
        effective_alpha = alpha if alpha is not None else self._config.get_regime_alpha(regime)
        accuracy = self._compute_accuracy(
            prediction_error, regime, series_id=series_id, engine_name=engine_name
        )

        # Record absolute error for per-engine sigma2_obs estimation
        self._variance_estimator.record_error(engine_name, prediction_error)
        sigma2_obs = self._variance_estimator.get_sigma2_obs(engine_name)

        previous_accuracy = self._accuracy[namespaced_regime].get(engine_name, 0.0)
        if engine_name not in self._priors[namespaced_regime]:
            self._priors[namespaced_regime][engine_name] = GaussianPrior(mu_0=accuracy, sigma2_0=1.0)
        observation = np.array([accuracy])
        posterior = self._bayesian.update(
            self._priors[namespaced_regime][engine_name],
            observation,
            sigma2_obs=sigma2_obs,
        )
        self._priors[namespaced_regime][engine_name] = posterior.to_prior()
        new_accuracy = posterior.get_param("mu_0", accuracy)
        if math.isfinite(new_accuracy):
            new_accuracy = float(max(0.0, min(1.0, new_accuracy)))
        else:
            new_accuracy = 0.0
        engine_names_in_regime = list(self._accuracy[namespaced_regime].keys())
        if engine_name not in engine_names_in_regime:
            engine_names_in_regime.append(engine_name)
        adaptive_lambda = compute_regularization_strength(
            self._update_counter,
            base_strength=self._regularization_strength,
            drift_score=drift_score if drift_score is not None else 0.0,
        )
        temp_accuracies = self._accuracy[namespaced_regime].copy()
        temp_accuracies[engine_name] = new_accuracy
        regularized = apply_l2_regularization(
            temp_accuracies,
            engine_names_in_regime,
            regularization_strength=adaptive_lambda,
        )
        final_accuracy = regularized[engine_name]
        if math.isfinite(final_accuracy):
            final_accuracy = float(max(0.0, min(1.0, final_accuracy)))
        else:
            final_accuracy = 0.0
        self._accuracy[namespaced_regime][engine_name] = final_accuracy
        if abs(new_accuracy - previous_accuracy) > self._config.immediate_persist_threshold:
            self.persist_immediately(engine_name, namespaced_regime)
        self._update_counter += 1
        if self._update_counter % _PERSIST_EVERY_N_UPDATES == 0:
            self._persistence.persist_regime_state(
                namespaced_regime, self._accuracy, self._priors,
                self._regime_last_access, self._regime_last_update,
            )
        self._redis.update_weight(namespaced_regime, engine_name, self._accuracy[namespaced_regime][engine_name])
