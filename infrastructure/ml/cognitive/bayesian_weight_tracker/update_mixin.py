"""Update mixin for BayesianWeightTracker."""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

import numpy as np

from iot_machine_learning.infrastructure.ml.inference.bayesian.prior import GaussianPrior

from .constants import _PERSIST_EVERY_N_UPDATES
from .per_sensor_key import build_regime_key
from .regularization import apply_l2_regularization, compute_regularization_strength

logger = logging.getLogger(__name__)


class UpdateMixin:
    """Mixin providing the Bayesian update cycle."""

    # Declare expected parent attributes for type checkers
    _lock: threading.Lock
    _domain_namespace: str
    _accuracy: dict[str, dict[str, float]]
    _config: Any
    _regime_last_access: dict[str, float]
    _regime_last_update: dict[str, float]
    _compute_accuracy: Any
    _variance_estimator: Any
    _priors: dict[str, dict[str, Any]]
    _estimate_data_variance: Any
    _prior_variance_scale: float
    _bayesian: Any
    _alpha: float
    _update_counter: int
    _regularization_strength: float
    _weight_history: dict[str, list[float]]
    _check_convergence: Any
    _dynamic_tuner: Any
    _persistence: Any
    _redis: Any
    persist_immediately: Any

    def update(
        self, regime: str, engine_name: str, prediction_error: float,
        alpha: float | None = None, *, series_id: str | None = None, drift_score: float | None = None,
    ) -> None:
        with self._lock:
            self._update_inner(
                regime, engine_name, prediction_error, alpha,
                series_id=series_id, drift_score=drift_score,
            )

    def _update_inner(
        self, regime: str, engine_name: str, prediction_error: float,
        alpha: float | None = None, *, series_id: str | None = None, drift_score: float | None = None,
    ) -> None:
        namespaced_regime = build_regime_key(self._domain_namespace, regime, series_id)
        if namespaced_regime not in self._accuracy and len(self._accuracy) >= self._config.max_regimes:
            coldest = min(self._regime_last_access, key=self._regime_last_access.__getitem__)
            del self._accuracy[coldest]
            del self._regime_last_access[coldest]
            self._regime_last_update.pop(coldest, None)
        now = time.monotonic()
        self._regime_last_access[namespaced_regime] = now
        self._regime_last_update[namespaced_regime] = now
        accuracy = self._compute_accuracy(
            prediction_error, regime, series_id=series_id, engine_name=engine_name
        )

        # Record absolute error for per-engine sigma2_obs estimation
        self._variance_estimator.record_error(engine_name, prediction_error)
        sigma2_obs = self._variance_estimator.get_sigma2_obs(engine_name)
        previous_accuracy = self._accuracy[namespaced_regime].get(engine_name, 0.0)

        # MATH-CRIT-2: Scale prior variance to data variance
        if engine_name not in self._priors[namespaced_regime]:
            data_var = self._estimate_data_variance(engine_name, min_samples=5, series_id=series_id)
            scaled_prior_var = self._prior_variance_scale * data_var if data_var is not None else 1.0
            self._priors[namespaced_regime][engine_name] = GaussianPrior(
                mu_0=accuracy, sigma2_0=scaled_prior_var,
            )

        observation = np.array([accuracy])
        posterior = self._bayesian.update(
            self._priors[namespaced_regime][engine_name], observation, sigma2_obs=sigma2_obs,
        )
        next_prior = posterior.to_prior()
        current_alpha = getattr(self, "_alpha", self._config.alpha)
        if current_alpha > 0 and sigma2_obs is not None:
            min_prior_var = 1.5 * current_alpha * float(sigma2_obs)
            if next_prior.parameters.get("sigma2_0", 0.0) < min_prior_var:
                next_prior.parameters["sigma2_0"] = min_prior_var
        self._priors[namespaced_regime][engine_name] = next_prior
        new_accuracy = posterior.get_param("mu_0", accuracy)
        new_accuracy = float(max(0.0, min(1.0, new_accuracy))) if math.isfinite(new_accuracy) else 0.0
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
            temp_accuracies, engine_names_in_regime, regularization_strength=adaptive_lambda,
        )
        final_accuracy = regularized[engine_name]
        final_accuracy = float(max(0.0, min(1.0, final_accuracy))) if math.isfinite(final_accuracy) else 0.0
        self._accuracy[namespaced_regime][engine_name] = final_accuracy

        # SEVERO-3: Track weight history and check convergence
        weight_key = f"{namespaced_regime}:{engine_name}"
        self._weight_history[weight_key].append(final_accuracy)

        if self._check_convergence(weight_key):
            # FASE-9: Use DynamicTuner if available for adaptive alpha tuning
            if self._dynamic_tuner:
                convergence_report = self._dynamic_tuner._convergence_detector.get_report("ML_BAYES_ALPHA")
                tuned_alpha = self._dynamic_tuner.tune_learning_rate(
                    name="ML_BAYES_ALPHA",
                    current_value=self._alpha,
                    performance_metric=final_accuracy,
                    convergence_report=convergence_report,
                )
                if tuned_alpha != self._alpha:
                    self._alpha = tuned_alpha
                    logger.info(
                        "bayesian_tracker_dynamic_tuning_applied",
                        extra={"regime": namespaced_regime, "engine": engine_name, "new_alpha": round(tuned_alpha, 4)},
                    )
            else:
                self._alpha *= 0.99
                logger.info(
                    "bayesian_tracker_convergence_detected",
                    extra={"regime": namespaced_regime, "engine": engine_name, "new_alpha": round(self._alpha, 4)},
                )

        if abs(new_accuracy - previous_accuracy) > self._config.immediate_persist_threshold:
            self.persist_immediately(engine_name, namespaced_regime)
        self._update_counter += 1
        if self._update_counter % _PERSIST_EVERY_N_UPDATES == 0:
            self._persistence.persist_regime_state(
                namespaced_regime, self._accuracy, self._priors,
                self._regime_last_access, self._regime_last_update,
            )
        self._redis.update_weight(namespaced_regime, engine_name, self._accuracy[namespaced_regime][engine_name])
