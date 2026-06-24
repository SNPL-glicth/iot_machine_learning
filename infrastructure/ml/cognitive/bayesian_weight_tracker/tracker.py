"""BayesianWeightTracker — consolidated main class.

Combines logic previously spread across: base.py, accuracy_mixin.py,
update_mixin.py, weights_mixin.py, reset_mixin.py, checkpoint_mixin.py.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np

from core.drift.drift_coupling import DriftNotifier, WeightTrackerDriftListener

from .config import BayesianWeightConfig, WeightTrackerConfig
from .updater import (
    BayesianUpdater,
    GaussianPrior,
    VarianceEstimator,
    apply_l2_regularization,
    build_fallback_key,
    build_regime_key,
    compute_accuracy,
    compute_regularization_strength,
    compute_weights_from_accuracy,
    should_use_per_sensor,
)
from .persistence import WeightTrackerCheckpoint, WeightTrackerPersistence, WeightTrackerRedisClient
from .drift_response import GradualDriftResponse

logger = logging.getLogger(__name__)


class BayesianWeightTracker:
    """Tracks per-regime, per-engine accuracy using Bayesian inference.

    Consolidates all weight tracking logic into a single class.
    """

    def __init__(
        self,
        config: Optional[BayesianWeightConfig] = None,
        repository: Optional[Any] = None,
        redis_client: Optional[Any] = None,
        use_redis: bool = False,
        scope: Optional[Any] = None,
        domain_namespace: str = "default",
        error_store: Optional[Any] = None,
        dynamic_tuner: Optional[Any] = None,
    ) -> None:
        self._config = config or BayesianWeightConfig()
        self._config.validate()
        self._domain_namespace = domain_namespace
        self._error_store = error_store
        self._scope = scope
        self._dynamic_tuner = dynamic_tuner

        self._drift_response = GradualDriftResponse(
            decay_factor=self._config.drift_decay_factor,
            variance_expansion=self._config.drift_variance_expansion,
        )

        # Core state
        self._accuracy: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._priors: Dict[str, Dict[str, GaussianPrior]] = defaultdict(dict)
        self._bayesian = BayesianUpdater()
        self._regime_last_access: Dict[str, float] = {}
        self._regime_last_update: Dict[str, float] = {}
        self._update_counter = 0
        self._error_history: Dict[str, List[float]] = defaultdict(list)
        self._weight_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._config.weight_history_maxlen)
        )
        self._alpha = self._config.alpha

        self._variance_estimator = VarianceEstimator(
            window=self._config.variance_window,
            min_samples=self._config.variance_min_samples,
            min_sigma2=self._config.sigma2_obs_min,
            default_sigma2=self._config.sigma2_obs_default,
        )

        self._lock = threading.RLock()
        self._persistence = WeightTrackerPersistence(repository)
        self._redis = WeightTrackerRedisClient(redis_client, scope)
        self._checkpoint = WeightTrackerCheckpoint(repository)

        # Warm start
        self._persistence.load_all_regimes(
            self._accuracy, self._priors,
            self._regime_last_access, self._regime_last_update,
        )

        drift_notifier = DriftNotifier()
        drift_notifier.subscribe(WeightTrackerDriftListener(self))

    # ── Public API ─────────────────────────────────────────────────

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
            self._update_inner(regime, engine_name, prediction_error, alpha,
                               series_id=series_id, drift_score=drift_score)

    def get_weights(
        self,
        regime: str,
        engine_names: List[str],
        series_id: Optional[str] = None,
    ) -> Dict[str, float]:
        with self._lock:
            if should_use_per_sensor(series_id, self._accuracy, self._domain_namespace, regime):
                key = build_regime_key(self._domain_namespace, regime, series_id)
            else:
                key = build_fallback_key(self._domain_namespace, regime)
            n = len(engine_names)
            if n == 0:
                return {}
            redis_w = self._redis.get_weights(key, engine_names, self._config.min_weight)
            if redis_w:
                return redis_w
            regime_data = self._accuracy.get(key, {})
            return compute_weights_from_accuracy(engine_names, regime_data, self._config.min_weight)

    def has_history(self, regime: str, series_id: Optional[str] = None) -> bool:
        with self._lock:
            key = build_regime_key(self._domain_namespace, regime, series_id)
            return bool(self._accuracy.get(key))

    def reset(self, regime: Optional[str] = None) -> None:
        with self._lock:
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
        with self._lock:
            namespaced = f"{self._domain_namespace}:{regime}"
            had_data = namespaced in self._accuracy
            if use_gradual_decay and had_data:
                decayed, expanded = self._drift_response.apply_decay(
                    self._accuracy[namespaced],
                    self._priors[namespaced],
                    drift_severity=drift_severity,
                )
                self._accuracy[namespaced] = decayed
                self._priors[namespaced] = expanded
                logger.warning(
                    "weights_gradual_decay",
                    extra={
                        "regime": regime,
                        "series_id": series_id or "unknown",
                        "decay": self._drift_response._decay,
                    },
                )
            else:
                self._accuracy.pop(namespaced, None)
                self._priors.pop(namespaced, None)
                self._regime_last_access.pop(namespaced, None)
                self._regime_last_update.pop(namespaced, None)

    def should_apply_decay(
        self, drift_magnitude: float, threshold: float = 0.7,
    ) -> bool:
        return self._drift_response.should_apply_decay(drift_magnitude, threshold)

    # ── Internal update logic ──────────────────────────────────────

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
        namespaced = build_regime_key(self._domain_namespace, regime, series_id)
        fallback_key = build_fallback_key(self._domain_namespace, regime)

        # LRU eviction
        if namespaced not in self._accuracy and len(self._accuracy) >= self._config.max_regimes:
            coldest = min(self._regime_last_access, key=self._regime_last_access.get)
            del self._accuracy[coldest]
            del self._regime_last_access[coldest]
            self._regime_last_update.pop(coldest, None)

        now = time.monotonic()
        self._regime_last_access[namespaced] = now
        self._regime_last_update[namespaced] = now

        effective_alpha = alpha if alpha is not None else self._config.get_regime_alpha(regime)

        accuracy = compute_accuracy(
            prediction_error,
            error_store=self._error_store,
            series_id=series_id,
            engine_name=engine_name,
            error_history=self._error_history,
        )

        self._variance_estimator.record_error(engine_name, prediction_error)
        sigma2_obs = self._variance_estimator.get_sigma2_obs(engine_name)

        prev_accuracy = self._accuracy[namespaced].get(engine_name, 0.0)

        if engine_name not in self._priors[namespaced]:
            self._priors[namespaced][engine_name] = GaussianPrior(
                mu_0=accuracy,
                sigma2_0=1.0,
            )

        obs = np.array([accuracy])
        posterior = self._bayesian.update(
            self._priors[namespaced][engine_name], obs, sigma2_obs=sigma2_obs,
        )
        self._priors[namespaced][engine_name] = posterior.to_prior()

        new_acc = posterior.mu_0
        if not math.isfinite(new_acc):
            new_acc = 0.0
        new_acc = float(max(0.0, min(1.0, new_acc)))

        engines_in_regime = list(self._accuracy[namespaced].keys())
        if engine_name not in engines_in_regime:
            engines_in_regime.append(engine_name)

        adaptive_lambda = compute_regularization_strength(
            self._update_counter,
            base_strength=self._config.regularization_strength,
            drift_score=drift_score if drift_score is not None else 0.0,
        )
        temp = dict(self._accuracy[namespaced])
        temp[engine_name] = new_acc
        regularized = apply_l2_regularization(temp, engines_in_regime, adaptive_lambda)
        final = regularized[engine_name]
        if not math.isfinite(final):
            final = 0.0
        final = float(max(0.0, min(1.0, final)))

        self._accuracy[namespaced][engine_name] = final

        # Convergence check
        wk = f"{namespaced}:{engine_name}"
        self._weight_history[wk].append(final)
        if self._check_convergence(wk):
            self._alpha *= 0.99

        # Persistence
        if abs(new_acc - prev_accuracy) > self._config.immediate_persist_threshold:
            self._persist_immediately(engine_name, namespaced)

        self._update_counter += 1
        if self._update_counter % 10 == 0:  # every 10 updates
            self._persistence.persist_regime_state(
                namespaced, self._accuracy, self._priors,
                self._regime_last_access, self._regime_last_update,
            )

        self._redis.update_weight(namespaced, engine_name, self._accuracy[namespaced][engine_name])

    def _check_convergence(self, weight_key: str) -> bool:
        history = self._weight_history.get(weight_key)
        if not history or len(history) < self._config.convergence_window:
            return False
        recent = list(history)[-self._config.convergence_window:]
        mu = sum(recent) / len(recent)
        if mu < 1e-12:
            return False
        var = sum((w - mu) ** 2 for w in recent) / len(recent)
        cv = math.sqrt(var) / mu
        return cv < self._config.convergence_cv_threshold

    def _persist_immediately(self, engine: str, regime_key: str) -> None:
        pass  # handled by periodic persistence

    # ── Natural decay: every 50 updates without feedback ───────────

    def apply_natural_decay(self, series_id: str, regime: str) -> None:
        """Reduce confidence by 5% after 50 predictions without feedback."""
        key = build_regime_key(self._domain_namespace, regime, series_id)
        with self._lock:
            if key not in self._accuracy:
                return
            for eng in self._accuracy[key]:
                self._accuracy[key][eng] *= 0.95
            now = time.time()
            self._regime_last_update[key] = now
            self._persistence.persist_regime_state(
                key, self._accuracy, self._priors,
                self._regime_last_access, self._regime_last_update,
            )
