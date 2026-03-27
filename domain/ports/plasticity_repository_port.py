"""PlasticityRepositoryPort — persistence contract for learned weights.

Hexagonal pattern: Domain defines the contract, infrastructure implements it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


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
    
    Implementations may use SQL Server, Redis, or file storage.
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
