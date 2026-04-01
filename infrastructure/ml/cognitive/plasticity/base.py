"""Regime-contextual weight learning (plasticity) with Bayesian updates.

Analogous to synaptic plasticity: if an engine consistently performs
well in a specific regime, its weight *in that regime* increases.

The tracker maintains a per-regime, per-engine accuracy history
and computes adaptive weights that reflect historical performance
within the current signal regime using Bayesian posterior updates.

Design:
    - In-memory with optional persistence (saves every N updates).
    - Redis-backed shared plasticity for multi-worker consistency.
    - Bayesian posterior updates instead of simple EMA.
    - Falls back to uniform weights when no history exists.

Pure logic — no I/O, no logging (repository handles persistence).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Any

from iot_machine_learning.infrastructure.ml.inference.bayesian.posterior import BayesianUpdater
from iot_machine_learning.infrastructure.ml.inference.bayesian.prior import GaussianPrior
from iot_machine_learning.domain.ports.plasticity_repository_port import (
    PlasticityRepositoryPort,
    RegimeWeightState,
)

logger = logging.getLogger(__name__)


# Exponential smoothing factor for accuracy updates
_ALPHA: float = 0.15

# Regime-specific alpha values for adaptive learning rates
_REGIME_ALPHA: Dict[str, float] = {
    "STABLE": 0.15,
    "TRENDING": 0.22,
    "VOLATILE": 0.45,
    "NOISY": 0.08,
    "TRANSITIONAL": 0.20,
}

# Minimum weight floor to prevent total suppression
_MIN_WEIGHT: float = 0.05

# Maximum regimes to track before LRU eviction
_MAX_REGIMES: int = 10

# Persist state every N updates (batching for performance)
_PERSIST_EVERY_N_UPDATES: int = 10

# Redis cache TTL for weights (seconds)
_REDIS_CACHE_TTL_SECONDS: float = 60.0


class PlasticityTracker:
    """Tracks per-regime, per-engine accuracy and computes adaptive weights using Bayesian updates.

    Attributes:
        _accuracy: Dict[regime][engine_name] → smoothed inverse error.
        _priors: Dict[regime][engine_name] → Bayesian priors for weight estimation.
        _bayesian: BayesianUpdater instance for posterior computation.
        _alpha: EMA smoothing factor (legacy, kept for compatibility).
        _min_weight: Minimum weight floor.
        _max_regimes: Maximum regimes before LRU eviction.
        _regime_last_access: Dict[regime] → monotonic timestamp for LRU.
        _repository: Optional repository for persistence.
        _update_counter: Counter for batching persistence.
    """

    def __init__(
        self,
        alpha: float = _ALPHA,
        min_weight: float = _MIN_WEIGHT,
        max_regimes: int = _MAX_REGIMES,
        regime_ttl_seconds: float = 86400.0,
        repository: Optional[PlasticityRepositoryPort] = None,
        redis_client: Optional[Any] = None,
        use_redis: bool = False,
    ) -> None:
        self._accuracy: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._priors: Dict[str, Dict[str, GaussianPrior]] = defaultdict(dict)
        self._bayesian = BayesianUpdater()
        self._alpha = alpha
        self._min_weight = min_weight
        self._max_regimes = max(1, max_regimes)
        self._regime_ttl_seconds = regime_ttl_seconds
        self._regime_last_access: Dict[str, float] = {}
        self._regime_last_update: Dict[str, float] = {}
        self._repository = repository
        self._update_counter = 0
        
        # Redis-backed shared plasticity
        self._redis = redis_client
        self._use_redis = use_redis and redis_client is not None
        self._local_cache: Dict[str, tuple[Dict[str, float], float]] = {}  # (weights, timestamp)
        
        # Load state from repository if available
        if self._repository is not None:
            self._load_all_regimes()

    def update(
        self,
        regime: str,
        engine_name: str,
        prediction_error: float,
        alpha: Optional[float] = None,
    ) -> None:
        """Record a prediction outcome for an engine in a regime.

        Args:
            regime: Current regime label.
            engine_name: Which engine made the prediction.
            prediction_error: |predicted - actual|.
            alpha: Override EMA smoothing factor for this call only.
                If None, uses self._alpha (default behaviour).
        """
        # Evict LRU regime if at capacity
        if regime not in self._accuracy and len(self._accuracy) >= self._max_regimes:
            coldest = min(self._regime_last_access, key=self._regime_last_access.get)
            del self._accuracy[coldest]
            del self._regime_last_access[coldest]
            self._regime_last_update.pop(coldest, None)
        
        now = time.monotonic()
        
        # TTL decay: exponential decay to equilibrium (uniform distribution)
        if regime in self._regime_last_update:
            elapsed = now - self._regime_last_update[regime]
            if elapsed > self._regime_ttl_seconds:
                import math
                n_engines = len(self._accuracy[regime])
                if n_engines > 0:
                    w_eq = 1.0 / n_engines
                    decay_rate = 1.0 / self._regime_ttl_seconds
                    decay_factor = math.exp(-decay_rate * elapsed)
                    
                    for eng in self._accuracy[regime]:
                        w_0 = self._accuracy[regime][eng]
                        w_t = w_0 * decay_factor + w_eq * (1.0 - decay_factor)
                        self._accuracy[regime][eng] = max(self._min_weight, min(1.0, w_t))
        
        # Update access and update times
        self._regime_last_access[regime] = now
        self._regime_last_update[regime] = now
        
        if alpha is not None:
            effective_alpha = alpha
        else:
            effective_alpha = _REGIME_ALPHA.get(regime, self._alpha)
        
        # Use Bayesian update instead of simple EMA
        inv_error = 1.0 / (abs(prediction_error) + 1e-9)
        
        # Initialize prior if this is first observation
        if engine_name not in self._priors[regime]:
            self._priors[regime][engine_name] = GaussianPrior(mu_0=inv_error, sigma2_0=1.0)
        
        # Bayesian update with single observation
        import numpy as np
        observation = np.array([inv_error])
        posterior = self._bayesian.update(self._priors[regime][engine_name], observation)
        
        # Store posterior as next prior and update accuracy
        self._priors[regime][engine_name] = posterior.to_prior()
        self._accuracy[regime][engine_name] = posterior.get_param("mu_0", inv_error)
        
        # Increment update counter and persist if needed
        self._update_counter += 1
        if (
            self._repository is not None
            and self._update_counter % _PERSIST_EVERY_N_UPDATES == 0
        ):
            self._persist_regime_state(regime)
        
        # Also update Redis if enabled (for real-time sharing across workers)
        if self._use_redis and self._redis is not None:
            self._update_redis(regime, engine_name, self._accuracy[regime][engine_name])

    def get_weights(
        self,
        regime: str,
        engine_names: List[str],
    ) -> Dict[str, float]:
        """Compute regime-contextual weights from accumulated accuracy.

        Args:
            regime: Current regime label.
            engine_names: List of engine names to weight.

        Returns:
            Dict[engine_name → weight], normalized to sum to 1.0.
            Falls back to uniform weights if no history for this regime.
        """
        n = len(engine_names)
        if n == 0:
            return {}
        
        # Try Redis first for shared weights (deterministic across workers)
        if self._use_redis and self._redis is not None:
            redis_weights = self._get_weights_from_redis(regime, engine_names)
            if redis_weights:
                return redis_weights
        
        # Fallback to local accuracy data
        regime_data = self._accuracy.get(regime, {})
        if not regime_data:
            uniform = 1.0 / n
            return {name: uniform for name in engine_names}
        
        now = time.monotonic()
        
        if regime in self._regime_last_update:
            elapsed = now - self._regime_last_update[regime]
            if elapsed > self._regime_ttl_seconds:
                import math
                w_eq = 1.0 / len(regime_data)
                decay_rate = 1.0 / self._regime_ttl_seconds
                decay_factor = math.exp(-decay_rate * elapsed)
                
                regime_data = {
                    eng: max(self._min_weight, min(1.0, 
                        w * decay_factor + w_eq * (1.0 - decay_factor)))
                    for eng, w in regime_data.items()
                }

        raw: Dict[str, float] = {}
        for name in engine_names:
            raw[name] = max(
                self._min_weight,
                regime_data.get(name, self._min_weight),
            )

        total = sum(raw.values())
        if total < 1e-12:
            uniform = 1.0 / n
            return {name: uniform for name in engine_names}

        return {name: w / total for name, w in raw.items()}
    
    def _get_weights_from_redis(
        self,
        regime: str,
        engine_names: List[str],
    ) -> Optional[Dict[str, float]]:
        """Fetch weights from Redis with local caching.
        
        Returns None if Redis unavailable or no data.
        """
        cache_key = f"plasticity:{regime}"
        now = time.monotonic()
        
        # Check local cache first
        if cache_key in self._local_cache:
            weights, timestamp = self._local_cache[cache_key]
            if now - timestamp < _REDIS_CACHE_TTL_SECONDS:
                return weights
        
        try:
            # Fetch from Redis
            redis_weights = self._redis.hgetall(cache_key)
            if not redis_weights:
                return None
            
            # Parse and filter to requested engines
            weights = {}
            for name in engine_names:
                if name in redis_weights:
                    weights[name] = float(redis_weights[name])
                else:
                    weights[name] = self._min_weight
            
            # Normalize
            total = sum(weights.values())
            if total < 1e-12:
                return None
            
            normalized = {name: w / total for name, w in weights.items()}
            
            # Cache locally
            self._local_cache[cache_key] = (normalized, now)
            
            return normalized
            
        except Exception as e:
            logger.debug(f"redis_weights_fetch_failed: {e}")
            return None

    def _update_redis(self, regime: str, engine_name: str, accuracy: float) -> None:
        """Update single engine weight in Redis (fail-safe)."""
        try:
            cache_key = f"plasticity:{regime}"
            self._redis.hset(cache_key, engine_name, str(accuracy))
            # Invalidate local cache to force refresh on next read
            self._local_cache.pop(cache_key, None)
        except Exception as e:
            logger.debug(f"redis_update_failed: {e}")

    def has_history(self, regime: str) -> bool:
        """True if any accuracy data exists for this regime."""
        return bool(self._accuracy.get(regime))

    def _load_all_regimes(self) -> None:
        """Load all regime states from repository on initialization.
        
        Private method called by __init__ when repository is provided.
        Fail-safe: logs warnings but doesn't raise on errors.
        """
        if self._repository is None:
            return
            
        try:
            regimes = self._repository.list_stored_regimes()
            for regime in regimes:
                # Get all engines for this regime
                states = self._repository.load_regime_state(regime, [])
                for key, state in states.items():
                    self._accuracy[state.regime][state.engine_name] = state.accuracy
                    self._priors[state.regime][state.engine_name] = GaussianPrior(
                        mu_0=state.prior_mu,
                        sigma2_0=state.prior_sigma2,
                    )
                    self._regime_last_access[state.regime] = state.last_access_time
                    self._regime_last_update[state.regime] = state.last_update_time
                    
            logger.debug(
                "plasticity_state_loaded",
                extra={"regimes_loaded": len(regimes), "total_engines": len(states)},
            )
        except Exception as e:
            logger.warning(
                "plasticity_load_all_failed",
                extra={"error": str(e)},
            )
            # Fail-safe: continue with empty state

    def _persist_regime_state(self, regime: str) -> None:
        """Persist current state for a regime to repository.
        
        Private method called periodically during updates.
        Fail-safe: logs warnings but doesn't raise on errors.
        """
        if self._repository is None:
            return
            
        if regime not in self._accuracy:
            return
            
        try:
            now = time.time()
            states = []
            for engine_name, accuracy in self._accuracy[regime].items():
                prior = self._priors[regime].get(engine_name)
                if prior is None:
                    continue
                    
                state = RegimeWeightState(
                    regime=regime,
                    engine_name=engine_name,
                    accuracy=accuracy,
                    prior_mu=getattr(prior, 'mu_0', accuracy),
                    prior_sigma2=getattr(prior, 'sigma2_0', 1.0),
                    last_access_time=self._regime_last_access.get(regime, now),
                    last_update_time=self._regime_last_update.get(regime, now),
                )
                states.append(state)
            
            if states:
                self._repository.save_regime_state(states)
                
        except Exception as e:
            logger.warning(
                "plasticity_persist_failed",
                extra={"regime": regime, "error": str(e)},
            )
            # Fail-safe: continue without persistence

    def reset(self, regime: Optional[str] = None) -> None:
        """Clear accumulated accuracy data.

        Args:
            regime: If provided, clear only that regime.
                If None, clear all regimes.
        """
        if regime is not None:
            self._accuracy.pop(regime, None)
        else:
            self._accuracy.clear()
