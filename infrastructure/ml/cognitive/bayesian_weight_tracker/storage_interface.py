"""Storage interface for posterior parameters (PERF-CRIT-2 DIP).

Applies DIP: BayesianWeightTracker depends on abstraction, not Redis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class IPosteriorStorage(ABC):
    """Interface for posterior parameter storage (PERF-CRIT-2).
    
    Allows BayesianWeightTracker to depend on abstraction, not Redis.
    
    Applies DIP: Concrete implementations (Redis, cache, mock) implement this.
    """
    
    @abstractmethod
    def load_posterior(
        self,
        regime: str,
        engine_name: str,
    ) -> Optional[Tuple[float, float]]:
        """Load posterior parameters.
        
        Args:
            regime: Regime key.
            engine_name: Engine name.
        
        Returns:
            Tuple of (mu, sigma2) if exists, else None.
        """
        pass
    
    @abstractmethod
    def save_posterior(
        self,
        regime: str,
        engine_name: str,
        mu: float,
        sigma2: float,
    ) -> None:
        """Save posterior parameters.
        
        Args:
            regime: Regime key.
            engine_name: Engine name.
            mu: Posterior mean.
            sigma2: Posterior variance.
        """
        pass
    
    @abstractmethod
    def delete_posterior(
        self,
        regime: str,
        engine_name: str,
    ) -> None:
        """Delete posterior parameters.
        
        Args:
            regime: Regime key.
            engine_name: Engine name.
        """
        pass
