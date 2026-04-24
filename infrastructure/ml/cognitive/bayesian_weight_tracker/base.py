"""Bayesian Weight Tracker — regime-contextual weight learning via Bayesian inference.

Tracks per-engine accuracy per regime using Bayesian updates with Gaussian priors.
NOT neuroplasticity or RL — honest naming for honest code.

GOLD version 0.2.2 — renamed from 'plasticity' to reflect actual implementation.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from iot_machine_learning.domain.ports.plasticity_repository_port import PlasticityRepositoryPort
from iot_machine_learning.domain.value_objects.plasticity_scope import PlasticityScope
from iot_machine_learning.infrastructure.ml.inference.bayesian.posterior import BayesianUpdater
from iot_machine_learning.infrastructure.ml.inference.bayesian.prior import GaussianPrior

from ..error_store import EngineErrorStore
from .constants import _PERSIST_EVERY_N_UPDATES, WeightTrackerConfig
from .redis_client import WeightTrackerRedisClient
from .persistence import WeightTrackerPersistence
from .checkpoint import WeightTrackerCheckpoint
from .weight_calculator import compute_weights_from_accuracy

_ERROR_STORE_PERCENTILE: float = 99.0
_ERROR_STORE_CAP_MULTIPLIER: float = 3.0
_ERROR_STORE_MIN_SAMPLES: int = 10
_PERCENTILE_SAMPLE_SIZE: int = 100

logger = logging.getLogger(__name__)


class BayesianWeightTracker:
    """Tracks per-regime, per-engine accuracy using Bayesian inference and computes adaptive weights."""
    
    def __init__(
        self,
        alpha: float = 0.15,
        min_weight: float = 0.05,
        max_regimes: int = 10,
        regime_ttl_seconds: float = 86400.0,
        repository: Optional[PlasticityRepositoryPort] = None,
        redis_client: Optional[Any] = None,
        use_redis: bool = False,
        scope: Optional[PlasticityScope] = None,
        immediate_persist_threshold: float = 0.15,
        domain_namespace: str = "default",
        error_store: Optional[EngineErrorStore] = None,
    ) -> None:
        self._config = WeightTrackerConfig(
            alpha=alpha, min_weight=min_weight, max_regimes=max_regimes,
            regime_ttl_seconds=regime_ttl_seconds,
            immediate_persist_threshold=immediate_persist_threshold,
        )
        self._scope = scope
        self._domain_namespace = domain_namespace
        self._error_store = error_store

        # Core state
        self._accuracy: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._priors: Dict[str, Dict[str, GaussianPrior]] = defaultdict(dict)
        self._bayesian = BayesianUpdater()
        self._regime_last_access: Dict[str, float] = {}
        self._regime_last_update: Dict[str, float] = {}
        self._update_counter = 0
        # Legacy in-memory error history. Only populated when no
        # ``error_store`` was injected; when a store is present it owns
        # the data and the local dict stays empty (IMP-4a).
        self._error_history: Dict[str, List[float]] = defaultdict(list)
        
        # Components
        self._persistence = WeightTrackerPersistence(repository)
        self._redis = WeightTrackerRedisClient(redis_client, scope)
        
        # Warm start
        self._persistence.load_all_regimes(
            self._accuracy, self._priors,
            self._regime_last_access, self._regime_last_update,
        )

    def _compute_accuracy(
        self,
        prediction_error: float,
        regime: str,
        series_id: Optional[str] = None,
        engine_name: Optional[str] = None,
    ) -> float:
        """Compute accuracy from prediction error.

        Caps the error at the 99th percentile of the relevant history
        times a safety multiplier, so a single outlier cannot destroy
        engine weights (ROBUSTNESS_AUDIT.md S7).

        IMP-4a: when an :class:`EngineErrorStore` is injected together
        with both ``series_id`` and ``engine_name`` the cap is computed
        from the store's history. The local ``_error_history`` dict is
        left untouched in that mode — the store is the single source
        of truth. Legacy callers (no store or missing identifiers) fall
        back to the previous per-regime list.
        """
        abs_error = abs(prediction_error)

        if (
            self._error_store is not None
            and series_id is not None
            and engine_name is not None
        ):
            recent = self._error_store.get_recent(
                series_id, engine_name, _PERCENTILE_SAMPLE_SIZE
            )
            if len(recent) >= _ERROR_STORE_MIN_SAMPLES:
                cap = self._error_store.get_percentile(
                    series_id, engine_name, _ERROR_STORE_PERCENTILE
                ) * _ERROR_STORE_CAP_MULTIPLIER
                if cap > 0.0:
                    abs_error = min(abs_error, cap)
            return 1.0 / (1.0 + abs_error)

        # Legacy per-regime history (no store injected)
        history = self._error_history.get(regime, [])
        if len(history) >= _ERROR_STORE_MIN_SAMPLES:
            cap = float(np.percentile(history, _ERROR_STORE_PERCENTILE)) * _ERROR_STORE_CAP_MULTIPLIER
            abs_error = min(abs_error, cap)

        if regime not in self._error_history:
            self._error_history[regime] = []
        self._error_history[regime].append(abs_error)
        if len(self._error_history[regime]) > _PERCENTILE_SAMPLE_SIZE:
            self._error_history[regime].pop(0)

        return 1.0 / (1.0 + abs_error)

    def update(
        self,
        regime: str,
        engine_name: str,
        prediction_error: float,
        alpha: Optional[float] = None,
        *,
        series_id: Optional[str] = None,
    ) -> None:
        """Record a prediction outcome.

        Args:
            regime: Signal regime label.
            engine_name: Prediction engine identifier.
            prediction_error: ``abs(predicted - actual)``.
            alpha: Optional override for the Bayesian learning rate.
            series_id: Optional series identifier. When supplied together
                with an injected :class:`EngineErrorStore`, the error is
                read from the store for percentile capping (IMP-4a).
        """
        # Add domain namespace to regime key
        namespaced_regime = f"{self._domain_namespace}:{regime}"

        # LRU eviction
        if namespaced_regime not in self._accuracy and len(self._accuracy) >= self._config.max_regimes:
            coldest = min(self._regime_last_access, key=self._regime_last_access.get)
            del self._accuracy[coldest]
            del self._regime_last_access[coldest]
            self._regime_last_update.pop(coldest, None)

        now = time.monotonic()
        self._regime_last_access[namespaced_regime] = now
        self._regime_last_update[namespaced_regime] = now

        # Compute accuracy (S7 fix: capped error; IMP-4a: store-aware cap)
        effective_alpha = alpha if alpha is not None else self._config.get_regime_alpha(regime)
        accuracy = self._compute_accuracy(
            prediction_error, regime, series_id=series_id, engine_name=engine_name
        )
        previous_accuracy = self._accuracy[namespaced_regime].get(engine_name, 0.0)
        
        # Bayesian update
        if engine_name not in self._priors[namespaced_regime]:
            self._priors[namespaced_regime][engine_name] = GaussianPrior(mu_0=accuracy, sigma2_0=1.0)
        
        observation = np.array([accuracy])
        posterior = self._bayesian.update(self._priors[namespaced_regime][engine_name], observation)
        self._priors[namespaced_regime][engine_name] = posterior.to_prior()
        new_accuracy = posterior.get_param("mu_0", accuracy)
        self._accuracy[namespaced_regime][engine_name] = new_accuracy
        
        # Immediate persistence for large changes
        if abs(new_accuracy - previous_accuracy) > self._config.immediate_persist_threshold:
            self.persist_immediately(engine_name, namespaced_regime)
        
        # Batch persistence
        self._update_counter += 1
        if self._update_counter % _PERSIST_EVERY_N_UPDATES == 0:
            self._persistence.persist_regime_state(
                namespaced_regime, self._accuracy, self._priors,
                self._regime_last_access, self._regime_last_update,
            )
        
        # Redis update
        self._redis.update_weight(namespaced_regime, engine_name, self._accuracy[namespaced_regime][engine_name])

    def get_weights(self, regime: str, engine_names: List[str]) -> Dict[str, float]:
        """Compute regime-contextual weights."""
        # Add domain namespace to regime key
        namespaced_regime = f"{self._domain_namespace}:{regime}"
        
        n = len(engine_names)
        if n == 0:
            return {}
        
        # Try Redis
        redis_weights = self._redis.get_weights(namespaced_regime, engine_names, self._config.min_weight)
        if redis_weights:
            return redis_weights
        
        # Local calculation
        regime_data = self._accuracy.get(namespaced_regime, {})
        return compute_weights_from_accuracy(engine_names, regime_data, self._config.min_weight)

    def has_history(self, regime: str) -> bool:
        """True if any accuracy data exists for this regime."""
        namespaced_regime = f"{self._domain_namespace}:{regime}"
        return bool(self._accuracy.get(namespaced_regime))

    def persist_immediately(self, engine_name: str, regime: str) -> None:
        """Immediately persist state."""
        self._persistence.persist_immediately(
            regime, engine_name, self._accuracy, self._priors,
            self._regime_last_access, self._regime_last_update,
        )

    def export_checkpoint(self) -> dict:
        """Export state as serializable checkpoint."""
        return WeightTrackerCheckpoint.export(
            self._scope, self._accuracy, self._priors,
            self._regime_last_access, self._regime_last_update,
            self._config.alpha, self._config.min_weight,
        )

    def restore_from_checkpoint(self, checkpoint_data: dict) -> None:
        """Restore state from checkpoint."""
        WeightTrackerCheckpoint.restore(
            checkpoint_data, self._scope, self._accuracy, self._priors,
            self._regime_last_access, self._regime_last_update,
        )

    def reset(self, regime: Optional[str] = None) -> None:
        """Clear accumulated accuracy data."""
        if regime is not None:
            self._accuracy.pop(regime, None)
        else:
            self._accuracy.clear()
