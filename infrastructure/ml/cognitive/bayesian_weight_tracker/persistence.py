"""Persistence operations for Bayesian weight tracker.

SQL and checkpoint persistence, isolated from core logic.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from iot_machine_learning.domain.ports.plasticity_repository_port import (
    PlasticityRepositoryPort,
    RegimeWeightState,
)
from iot_machine_learning.infrastructure.ml.inference.bayesian.prior import GaussianPrior

logger = logging.getLogger(__name__)


class WeightTrackerPersistence:
    """Persistence operations for weight tracking state.
    
    Responsibilities:
    - Load state from repository on initialization
    - Persist state periodically (batch)
    - Immediate persistence for critical changes
    """
    
    def __init__(
        self,
        repository: Optional[PlasticityRepositoryPort] = None,
    ) -> None:
        # Repository is mandatory - use InMemory if none provided
        if repository is None:
            from iot_machine_learning.infrastructure.persistence.inmemory.plasticity_repository import (
                InMemoryPlasticityRepository,
            )
            self._repository: PlasticityRepositoryPort = InMemoryPlasticityRepository()
        else:
            self._repository = repository
    
    @property
    def repository(self) -> PlasticityRepositoryPort:
        """Access the underlying repository."""
        return self._repository
    
    def load_all_regimes(
        self,
        accuracy: Dict[str, Dict[str, float]],
        priors: Dict[str, Dict[str, GaussianPrior]],
        regime_last_access: Dict[str, float],
        regime_last_update: Dict[str, float],
    ) -> None:
        """Load all regime states from repository.
        
        Populates the provided dictionaries in-place.
        Fail-safe: logs warnings but doesn't raise on errors.
        """
        try:
            regimes = self._repository.list_stored_regimes()
            for regime in regimes:
                states = self._repository.load_regime_state(regime, [])
                for key, state in states.items():
                    accuracy[state.regime][state.engine_name] = state.accuracy
                    priors[state.regime][state.engine_name] = GaussianPrior(
                        mu_0=state.prior_mu,
                        sigma2_0=state.prior_sigma2,
                    )
                    regime_last_access[state.regime] = state.last_access_time
                    regime_last_update[state.regime] = state.last_update_time
            
            logger.debug(
                "plasticity_state_loaded",
                extra={"regimes_loaded": len(regimes), "total_engines": len(states) if regimes else 0},
            )
        except Exception as e:
            logger.warning("plasticity_load_all_failed", extra={"error": str(e)})
    
    def persist_regime_state(
        self,
        regime: str,
        accuracy: Dict[str, Dict[str, float]],
        priors: Dict[str, Dict[str, GaussianPrior]],
        regime_last_access: Dict[str, float],
        regime_last_update: Dict[str, float],
    ) -> None:
        """Persist current state for a regime to repository.
        
        Batch persistence - called periodically.
        """
        if regime not in accuracy:
            return
        
        try:
            now = time.time()
            states = []
            for engine_name, acc in accuracy[regime].items():
                prior = priors[regime].get(engine_name)
                if prior is None:
                    continue
                
                state = RegimeWeightState(
                    regime=regime,
                    engine_name=engine_name,
                    accuracy=acc,
                    prior_mu=getattr(prior, 'mu_0', acc),
                    prior_sigma2=getattr(prior, 'sigma2_0', 1.0),
                    last_access_time=regime_last_access.get(regime, now),
                    last_update_time=regime_last_update.get(regime, now),
                )
                states.append(state)
            
            if states:
                self._repository.save_regime_state(states)
        except Exception as e:
            logger.warning("plasticity_persist_failed", extra={"regime": regime, "error": str(e)})
    
    def persist_immediately(
        self,
        regime: str,
        engine_name: str,
        accuracy: Dict[str, Dict[str, float]],
        priors: Dict[str, Dict[str, GaussianPrior]],
        regime_last_access: Dict[str, float],
        regime_last_update: Dict[str, float],
    ) -> None:
        """Immediately persist state for a specific engine and regime.
        
        Used when accuracy changes significantly.
        """
        if regime not in accuracy or engine_name not in accuracy[regime]:
            return
        
        try:
            now = time.time()
            acc = accuracy[regime][engine_name]
            prior = priors[regime].get(engine_name)
            
            if prior is None:
                return
            
            state = RegimeWeightState(
                regime=regime,
                engine_name=engine_name,
                accuracy=acc,
                prior_mu=getattr(prior, 'mu_0', acc),
                prior_sigma2=getattr(prior, 'sigma2_0', 1.0),
                last_access_time=regime_last_access.get(regime, now),
                last_update_time=regime_last_update.get(regime, now),
            )
            self._repository.save_regime_state([state])
            
            logger.debug(
                "plasticity_persist_immediate",
                extra={"regime": regime, "engine": engine_name, "accuracy": acc},
            )
        except Exception as e:
            logger.warning(
                "plasticity_persist_immediate_failed",
                extra={"regime": regime, "engine": engine_name, "error": str(e)},
            )
