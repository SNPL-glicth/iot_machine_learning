"""Bayesian Weight Tracker — regime-contextual weight learning via Bayesian inference.

Tracks per-engine accuracy per regime using Bayesian updates with Gaussian priors.
NOT neuroplasticity or RL — honest naming for honest code.

GOLD version 0.2.2 — renamed from 'plasticity' to reflect actual implementation.
"""

from __future__ import annotations

import logging
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
from .drift_response import GradualDriftResponse
from .variance_estimator import VarianceEstimator
from .accuracy_mixin import AccuracyMixin
from .update_mixin import UpdateMixin
from .weights_mixin import WeightsMixin
from .checkpoint_mixin import CheckpointMixin
from .reset_mixin import ResetMixin

logger = logging.getLogger(__name__)


class BayesianWeightTracker(
    AccuracyMixin,
    UpdateMixin,
    WeightsMixin,
    CheckpointMixin,
    ResetMixin,
):
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
        regularization_strength: float = 0.01,
        drift_decay_factor: float = 0.5,
        drift_variance_expansion: float = 2.0,
        sigma2_obs_default: float = 1.0,
        sigma2_obs_min: float = 0.01,
        variance_window: int = 20,
        variance_min_samples: int = 5,
    ) -> None:
        self._config = WeightTrackerConfig(
            alpha=alpha, min_weight=min_weight, max_regimes=max_regimes,
            regime_ttl_seconds=regime_ttl_seconds,
            immediate_persist_threshold=immediate_persist_threshold,
            sigma2_obs_default=sigma2_obs_default,
            sigma2_obs_min=sigma2_obs_min,
            variance_window=variance_window,
            variance_min_samples=variance_min_samples,
        )
        self._scope = scope
        self._domain_namespace = domain_namespace
        self._error_store = error_store
        self._regularization_strength = regularization_strength
        self._drift_response = GradualDriftResponse(
            decay_factor=drift_decay_factor,
            variance_expansion=drift_variance_expansion,
        )

        # Core state
        self._accuracy: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._priors: Dict[str, Dict[str, GaussianPrior]] = defaultdict(dict)
        self._bayesian = BayesianUpdater()
        self._regime_last_access: Dict[str, float] = {}
        self._regime_last_update: Dict[str, float] = {}
        self._update_counter = 0
        # Legacy in-memory error history.
        self._error_history: Dict[str, List[float]] = defaultdict(list)

        # Per-engine online variance estimator for Bayesian observation sigma2
        self._variance_estimator = VarianceEstimator(
            window=self._config.variance_window,
            min_samples=self._config.variance_min_samples,
            min_sigma2=self._config.sigma2_obs_min,
            default_sigma2=self._config.sigma2_obs_default,
        )

        # Thread safety for concurrent updates
        import threading
        self._lock = threading.RLock()

        # Components
        self._persistence = WeightTrackerPersistence(repository)
        self._redis = WeightTrackerRedisClient(redis_client, scope)

        # Warm start
        self._persistence.load_all_regimes(
            self._accuracy, self._priors,
            self._regime_last_access, self._regime_last_update,
        )
