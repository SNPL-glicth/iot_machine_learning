from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..entities.plasticity.signal_context import SignalContext


class PlasticityPort(ABC):
    """Contract for regime-contextual weight learning.

    Single source of truth for engine weight adaptation.  Infrastructure
    implementations may use EMA, contextual MAE, or any combination.
    """

    @abstractmethod
    def get_weights(
        self,
        regime: str,
        engine_names: List[str],
        context: Optional[SignalContext] = None,
    ) -> Dict[str, float]:
        """Return normalized weights for the given engines in this regime.

        Args:
            regime: Current signal regime label.
            engine_names: Names of the engines to weight.
            context: Optional richer context for contextual implementations.

        Returns:
            Dict[engine_name → weight], normalized to sum to 1.0.
            Falls back to uniform weights when no history exists.
        """
        ...

    @abstractmethod
    def record_error(
        self,
        engine_name: str,
        error: float,
        regime: str,
        context: Optional[SignalContext] = None,
    ) -> None:
        """Record a prediction error for an engine in a regime.

        Args:
            engine_name: Engine that produced the prediction.
            error: |predicted - actual|.
            regime: Signal regime at the time of prediction.
            context: Optional richer context for contextual implementations.
        """
        ...

    def has_history(self, regime: str) -> bool:
        """Return True if any learning data exists for this regime.

        Default implementation always returns False.  Implementations that
        maintain history should override this.
        """
        return False
