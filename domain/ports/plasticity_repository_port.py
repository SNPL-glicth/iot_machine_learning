"""PlasticityRepositoryPort — persistence contract for learned weights.

Hexagonal pattern: Domain defines the contract, infrastructure implements it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeWeightState:
    """Serializable state for a single regime-engine pair.
    
    Attributes:
        regime: Regime label (e.g., "STABLE", "TRENDING")
        engine_name: Engine identifier (e.g., "taylor", "baseline")
        accuracy: Smoothed inverse error (weight proxy)
        prior_mu: Bayesian prior mean
        prior_sigma2: Bayesian prior variance
        last_access_time: Unix timestamp of last access
        last_update_time: Unix timestamp of last update
    """
    regime: str
    engine_name: str
    accuracy: float
    prior_mu: float
    prior_sigma2: float
    last_access_time: float
    last_update_time: float


class PlasticityRepositoryPort(ABC):
    """Contract for persisting and retrieving plasticity state.
    
    Implementations may use SQL Server, key-value stores, or file storage.
    All methods are fail-safe: errors are logged but not raised
    to the domain layer.
    
    Usage:
        repo = SqlPlasticityRepository(connection)
        tracker = PlasticityTracker(repository=repo)
        # State loads automatically on init, saves every N updates
    """

    @abstractmethod
    def load_regime_state(
        self,
        regime: str,
        engine_names: List[str],
    ) -> Dict[str, RegimeWeightState]:
        """Load saved state for a regime and list of engines.
        
        Args:
            regime: Regime label to load
            engine_names: List of engine names to load state for
            
        Returns:
            Dict mapping (regime, engine_name) -> RegimeWeightState
            Only returns entries that exist in storage.
        """
        ...

    @abstractmethod
    def save_regime_state(
        self,
        states: List[RegimeWeightState],
    ) -> None:
        """Persist regime-engine states.
        
        Args:
            states: List of states to save (batch operation)
            
        Implementation should use UPSERT/MERGE to handle
            both inserts and updates.
        """
        ...

    @abstractmethod
    def list_stored_regimes(self) -> List[str]:
        """Return list of all regimes with stored state.
        
        Used to warm-cache all historical regimes on startup.
        """
        ...

    def has_regime_state(self, regime: str) -> bool:
        """Check if any state exists for this regime.
        
        Default implementation lists all regimes and checks.
        Implementations may override for efficiency.
        """
        return regime in self.list_stored_regimes()


class InMemoryPlasticityRepository(PlasticityRepositoryPort):
    """In-memory storage for plasticity state.
    
    WARNING: State is lost on process restart. Use only for development
    or when persistent storage is not configured.
    """
    
    def __init__(self, warn_on_init: bool = False) -> None:
        self._state: Dict[str, List[RegimeWeightState]] = {}
        if warn_on_init:
            logger.debug("InMemoryPlasticityRepository initialized. State will be LOST on restart.")
    
    def load_regime_state(
        self,
        regime: str,
        engine_names: List[str],
    ) -> Dict[str, RegimeWeightState]:
        result: Dict[str, RegimeWeightState] = {}
        states = self._state.get(regime, [])
        for state in states:
            if not engine_names or state.engine_name in engine_names:
                key = f"{state.regime}|{state.engine_name}"
                result[key] = state
        return result
    
    def save_regime_state(
        self,
        states: List[RegimeWeightState],
    ) -> None:
        if not states:
            return
        for state in states:
            if state.regime not in self._state:
                self._state[state.regime] = []
            existing = None
            for i, s in enumerate(self._state[state.regime]):
                if s.engine_name == state.engine_name:
                    existing = i
                    break
            if existing is not None:
                self._state[state.regime][existing] = state
            else:
                self._state[state.regime].append(state)
    
    def list_stored_regimes(self) -> List[str]:
        return list(self._state.keys())
    
    def clear(self) -> None:
        self._state.clear()
    
    def export_all(self) -> Dict[str, List[Dict]]:
        return {
            regime: [
                {
                    "regime": s.regime,
                    "engine_name": s.engine_name,
                    "accuracy": s.accuracy,
                    "prior_mu": s.prior_mu,
                    "prior_sigma2": s.prior_sigma2,
                    "last_access_time": s.last_access_time,
                    "last_update_time": s.last_update_time,
                }
                for s in states
            ]
            for regime, states in self._state.items()
        }
