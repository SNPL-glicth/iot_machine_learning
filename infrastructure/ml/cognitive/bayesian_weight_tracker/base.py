"""Bayesian Weight Tracker — regime-contextual weight learning via Bayesian inference.

Tracks per-engine accuracy per regime using Bayesian updates with Gaussian priors.
NOT neuroplasticity or RL — honest naming for honest code.

GOLD version 0.2.2 — renamed from 'plasticity' to reflect actual implementation.

FASE-21: Interacción con Inhibition Gate
-----------------------------------------
Bayesian Weight Tracker opera ANTES de Inhibition Gate en el pipeline.
Calcula pesos basados en historial de accuracy, luego Inhibition puede
suprimirlos basándose en señales hard (stability, fit_error, recent_error).

Resolución de conflictos:
- **Bayesian tiene precedencia** en ajuste gradual (señales soft):
  * Actualización incremental basada en historial de accuracy
  * EMA con alpha=0.15 (convergencia suave, no oscilante)
  * Ajuste fino en condiciones normales

- **Inhibition tiene precedencia** cuando detecta señales hard:
  * Supresión inmediata si stability/error exceden thresholds
  * Actúa como circuit breaker para casos extremos
  * Puede reducir a 0.0 pesos que Bayesian está promoviendo

La combinación es estable porque ambos sistemas son convergentes
(Bayesian EMA + Inhibition exponential decay = no oscilación).

Ver también: `infrastructure/ml/cognitive/inhibition/gate.py`
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from core.drift.drift_coupling import DriftNotifier, WeightTrackerDriftListener
from core.parameters.numerical_constants import EPSILON
from core.tuning.dynamic_tuning import DynamicTuner

from iot_machine_learning.domain.ports.plasticity_repository_port import PlasticityRepositoryPort
from iot_machine_learning.domain.value_objects.plasticity_scope import PlasticityScope
from iot_machine_learning.infrastructure.ml.inference.bayesian.posterior import BayesianUpdater
from iot_machine_learning.infrastructure.ml.inference.bayesian.prior import GaussianPrior

from ..error_store import EngineErrorStore
from .bayesian_weight_config import BayesianWeightConfig
from .constants import _PERSIST_EVERY_N_UPDATES
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
        config: Optional[BayesianWeightConfig] = None,
        repository: Optional[PlasticityRepositoryPort] = None,
        redis_client: Optional[Any] = None,
        use_redis: bool = False,
        scope: Optional[PlasticityScope] = None,
        domain_namespace: str = "default",
        error_store: Optional[EngineErrorStore] = None,
        dynamic_tuner: Optional[DynamicTuner] = None,
    ) -> None:
        """Initialize Bayesian weight tracker with injectable configuration.
        
        Args:
            config: BayesianWeightConfig with all parameters.
                   Defaults to standard config if not provided.
            repository: Plasticity repository for persistence.
            redis_client: Redis client for caching.
            use_redis: Whether to use Redis caching.
            scope: Plasticity scope for namespacing.
            domain_namespace: Domain namespace for multi-tenant support.
            error_store: Engine error store for variance estimation.
        
        Applies DIP: Configuration is injected, not read from literals.
        """
        self._config = config or BayesianWeightConfig()
        self._config.validate()  # Fail fast on invalid config
        
        self._scope = scope
        self._domain_namespace = domain_namespace
        self._error_store = error_store
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
        # Legacy in-memory error history.
        self._error_history: Dict[str, List[float]] = defaultdict(list)
        
        # SEVERO-3: Weight history for convergence detection
        from collections import deque
        self._weight_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._config.weight_history_maxlen)
        )
        self._alpha = self._config.alpha  # Store for decay on convergence

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
        self._dynamic_tuner = dynamic_tuner  # FASE-9: Optional dynamic tuning

        # Warm start
        self._persistence.load_all_regimes(
            self._accuracy, self._priors,
            self._regime_last_access, self._regime_last_update,
        )
        
        # NUEVO: Suscribir a drift events
        drift_notifier = DriftNotifier()
        drift_notifier.subscribe(WeightTrackerDriftListener(self))
    
    def _check_convergence(self, weight_key: str) -> bool:
        """Check if weights have converged (SEVERO-3).
        
        Args:
            weight_key: Key for weight history (regime:engine_name).
        
        Returns:
            True if coefficient of variation of last 10 weights < 0.05.
        
        Applies SRP: Convergence check is independent of update logic.
        """
        history = self._weight_history.get(weight_key)
        
        if not history or len(history) < self._config.convergence_window:
            return False
        
        # Get last N weights
        recent_weights = list(history)[-self._config.convergence_window:]
        
        # Calculate coefficient of variation
        mean_weight = sum(recent_weights) / len(recent_weights)
        
        if mean_weight < EPSILON.COMPARISON:  # Avoid division by zero
            return False
        
        variance = sum((w - mean_weight) ** 2 for w in recent_weights) / len(recent_weights)
        std_dev = variance ** 0.5
        cv = std_dev / mean_weight
        
        # Converged if CV < threshold
        return cv < self._config.convergence_cv_threshold
    
    def _estimate_data_variance(self, engine_name: str, min_samples: int = 5) -> Optional[float]:
        """Estimate data variance from recent errors (MATH-CRIT-2).
        
        Args:
            engine_name: Engine to estimate variance for.
            min_samples: Minimum samples required for estimation.
        
        Returns:
            Estimated variance or None if insufficient data.
        
        Applies DIP: Abstract method for variance estimation, mockeable in tests.
        """
        if self._error_store is None:
            return None
        
        try:
            # Get recent errors from error store
            recent_errors = self._error_store.get_recent(
                series_id="*",  # Aggregate across series for engine-level variance
                engine_name=engine_name,
                n=self._config.variance_window,
            )
            
            if len(recent_errors) < min_samples:
                return None
            
            # Compute empirical variance
            errors_array = np.array(recent_errors)
            variance = float(np.var(errors_array))
            
            # Ensure positive and finite
            if not np.isfinite(variance) or variance <= 0:
                return None
            
            return variance
        
        except Exception:
            # Fallback: no variance estimate
            return None
