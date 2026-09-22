"""Storage interface for posterior parameters and weight cache (PERF-CRIT-2 DIP).

Applies DIP: BayesianWeightTracker depends on abstraction, not concrete Redis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable


class IPosteriorStorage(ABC):
    """Interface for posterior parameter storage (PERF-CRIT-2).
    
    Allows BayesianWeightTracker to depend on abstraction, not Redis.
    
    Applies DIP: Concrete implementations implement this.
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


@runtime_checkable
class IWeightCache(Protocol):
    """Protocol for weight cache abstraction to decouple ML math from Redis."""

    def get_weights(
        self,
        regime: str,
        engine_names: List[str],
        min_weight: float,
    ) -> Optional[Dict[str, float]]:
        """Fetch cached weights."""
        ...

    def update_weight(
        self,
        regime: str,
        engine_name: str,
        accuracy: float,
    ) -> None:
        """Update single engine weight."""
        ...

    def update_weights_batch(
        self,
        regime: str,
        engine_accuracies: Dict[str, float],
    ) -> None:
        """Update multiple engine weights."""
        ...

    def invalidate_cache(self, regime: str) -> None:
        """Invalidate cache for regime."""
        ...


class InMemoryWeightCache(IWeightCache):
    """In-memory implementation of IWeightCache."""

    def __init__(self) -> None:
        self._weights: Dict[str, Dict[str, float]] = {}

    def get_weights(
        self,
        regime: str,
        engine_names: List[str],
        min_weight: float,
    ) -> Optional[Dict[str, float]]:
        if regime not in self._weights:
            return None
        weights = {}
        for name in engine_names:
            weights[name] = self._weights[regime].get(name, min_weight)
        total = sum(weights.values())
        if total < 1e-12:
            return None
        return {k: v / total for k, v in weights.items()}

    def update_weight(
        self,
        regime: str,
        engine_name: str,
        accuracy: float,
    ) -> None:
        if regime not in self._weights:
            self._weights[regime] = {}
        self._weights[regime][engine_name] = accuracy

    def update_weights_batch(
        self,
        regime: str,
        engine_accuracies: Dict[str, float],
    ) -> None:
        if regime not in self._weights:
            self._weights[regime] = {}
        self._weights[regime].update(engine_accuracies)

    def invalidate_cache(self, regime: str) -> None:
        self._weights.pop(regime, None)
