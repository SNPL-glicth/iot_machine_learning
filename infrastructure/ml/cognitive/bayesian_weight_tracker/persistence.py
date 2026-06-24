"""Persistence operations for BayesianWeightTracker.

Consolidates: persistence.py, redis_client.py, checkpoint.py,
storage_interface.py, cached_storage.py, posterior_cache.py,
error_persister.py.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from iot_machine_learning.domain.ports.plasticity_repository_port import (
    PlasticityRepositoryPort,
    RegimeWeightState,
)

from .updater import GaussianPrior
from .config import _REDIS_CACHE_TTL_SECONDS, _PERSIST_EVERY_N_UPDATES

logger = logging.getLogger(__name__)


# ── WeightTrackerRedisClient ────────────────────────────────────────


class WeightTrackerRedisClient:
    """Redis caching for weight tracker state."""

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        scope: Optional[Any] = None,
    ) -> None:
        self._redis = redis_client
        self._scope = scope

    def get_weights(
        self,
        regime_key: str,
        engine_names: List[str],
        min_weight: float,
    ) -> Optional[Dict[str, float]]:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(f"bwt:w:{regime_key}")
            if raw is None:
                return None
            data = json.loads(raw)
            return {e: max(min_weight, float(data.get(e, min_weight))) for e in engine_names}
        except Exception:
            return None

    def update_weight(self, regime_key: str, engine: str, weight: float) -> None:
        if self._redis is None:
            return
        try:
            key = f"bwt:w:{regime_key}"
            raw = self._redis.get(key)
            if raw:
                data = json.loads(raw)
            else:
                data = {}
            data[engine] = weight
            self._redis.setex(key, int(_REDIS_CACHE_TTL_SECONDS), json.dumps(data))
        except Exception:
            pass


# ── WeightTrackerPersistence ────────────────────────────────────────


class WeightTrackerPersistence:
    """SQL/repository persistence for weight tracker state."""

    def __init__(self, repository: Optional[PlasticityRepositoryPort] = None) -> None:
        if repository is None:
            from iot_machine_learning.infrastructure.persistence.inmemory.plasticity_repository import (
                InMemoryPlasticityRepository,
            )
            self._repository: PlasticityRepositoryPort = InMemoryPlasticityRepository()
        else:
            self._repository = repository

    @property
    def repository(self) -> PlasticityRepositoryPort:
        return self._repository

    def load_all_regimes(
        self,
        accuracy: Dict[str, Dict[str, float]],
        priors: Dict[str, Dict[str, GaussianPrior]],
        regime_last_access: Dict[str, float],
        regime_last_update: Dict[str, float],
    ) -> None:
        try:
            regimes = self._repository.list_stored_regimes()
            states: Dict[str, RegimeWeightState] = {}
            for regime in regimes:
                rs = self._repository.load_regime_state(regime, [])
                for k, s in rs.items():
                    states[k] = s
                    accuracy[s.regime][s.engine_name] = s.accuracy
                    priors[s.regime][s.engine_name] = GaussianPrior(
                        mu_0=s.prior_mu, sigma2_0=s.prior_sigma2,
                    )
                    regime_last_access[s.regime] = s.last_access_time
                    regime_last_update[s.regime] = s.last_update_time
            logger.debug(
                "state_loaded",
                extra={"regimes": len(regimes), "engines": len(states)},
            )
        except Exception as e:
            logger.warning(f"load_all_failed: {e}")

    def persist_regime_state(
        self,
        regime: str,
        accuracy: Dict[str, Dict[str, float]],
        priors: Dict[str, Dict[str, GaussianPrior]],
        regime_last_access: Dict[str, float],
        regime_last_update: Dict[str, float],
    ) -> None:
        if regime not in accuracy:
            return
        try:
            now = time.time()
            states = []
            for engine, acc in accuracy[regime].items():
                prior = priors[regime].get(engine)
                if prior is None:
                    continue
                states.append(RegimeWeightState(
                    regime=regime,
                    engine_name=engine,
                    accuracy=acc,
                    prior_mu=prior.mu_0,
                    prior_sigma2=prior.sigma2_0,
                    last_access_time=regime_last_access.get(regime, now),
                    last_update_time=regime_last_update.get(regime, now),
                ))
            if states:
                self._repository.save_regime_state(states)
        except Exception as e:
            logger.warning(f"persist_failed regime={regime}: {e}")


# ── WeightTrackerCheckpoint ─────────────────────────────────────────


class WeightTrackerCheckpoint:
    """Export/import checkpoint for gossip protocol."""

    def __init__(self, repository: Optional[PlasticityRepositoryPort] = None) -> None:
        self._persistence = WeightTrackerPersistence(repository)

    def export_state(
        self,
        accuracy: Dict[str, Dict[str, float]],
        priors: Dict[str, Dict[str, GaussianPrior]],
        regime_last_access: Dict[str, float],
        regime_last_update: Dict[str, float],
    ) -> Dict[str, Any]:
        now = time.time()
        checkpoint: Dict[str, Any] = {"regimes": {}, "timestamp": now}
        for regime, engines in accuracy.items():
            checkpoint["regimes"][regime] = {}
            for ename, acc in engines.items():
                prior = priors.get(regime, {}).get(ename, GaussianPrior(0.0, 1.0))
                checkpoint["regimes"][regime][ename] = {
                    "accuracy": acc,
                    "prior_mu": prior.mu_0,
                    "prior_sigma2": prior.sigma2_0,
                    "last_access": regime_last_access.get(regime, now),
                    "last_update": regime_last_update.get(regime, now),
                }
        return checkpoint

    def import_state(
        self,
        checkpoint: Dict[str, Any],
        accuracy: Dict[str, Dict[str, float]],
        priors: Dict[str, Dict[str, GaussianPrior]],
        regime_last_access: Dict[str, float],
        regime_last_update: Dict[str, float],
    ) -> None:
        ts = checkpoint.get("timestamp", time.time())
        for regime, engines in checkpoint.get("regimes", {}).items():
            for ename, data in engines.items():
                accuracy.setdefault(regime, {})[ename] = data.get("accuracy", 0.5)
                priors.setdefault(regime, {})[ename] = GaussianPrior(
                    mu_0=data.get("prior_mu", 0.5),
                    sigma2_0=data.get("prior_sigma2", 1.0),
                )
                regime_last_access.setdefault(regime, ts)
                regime_last_update.setdefault(regime, ts)
