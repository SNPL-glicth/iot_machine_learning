"""In-memory implementation of PlasticityRepositoryPort.

Used as fallback when no persistent repository is configured.
Warns on initialization to alert operators about state loss on restart.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from iot_machine_learning.domain.ports.plasticity_repository_port import (
    PlasticityRepositoryPort,
    RegimeWeightState,
)

logger = logging.getLogger(__name__)


class InMemoryPlasticityRepository(PlasticityRepositoryPort):
    """In-memory storage for plasticity state.
    
    WARNING: State is lost on process restart. Use only for development
    or when SQL persistence is temporarily unavailable.
    
    Args:
        warn_on_init: If True, logs a warning on initialization.
    """
    
    def __init__(self, warn_on_init: bool = True) -> None:
        self._state: Dict[str, List[RegimeWeightState]] = {}
        
        if warn_on_init:
            logger.warning(
                "InMemoryPlasticityRepository initialized. "
                "Plasticity state will be LOST on process restart. "
                "Configure SqlPlasticityRepository for production."
            )
    
    def load_regime_state(
        self,
        regime: str,
        engine_names: List[str],
    ) -> Dict[str, RegimeWeightState]:
        """Load saved state for a regime and list of engines.
        
        Returns empty dict if no state found.
        """
        result: Dict[str, RegimeWeightState] = {}
        
        states = self._state.get(regime, [])
        for state in states:
            if not engine_names or state.engine_name in engine_names:
                key = f"{state.regime}|{state.engine_name}"
                result[key] = state
        
        logger.debug(
            "inmemory_plasticity_loaded",
            extra={"regime": regime, "engines_found": len(result)},
        )
        return result
    
    def save_regime_state(
        self,
        states: List[RegimeWeightState],
    ) -> None:
        """Persist regime-engine states in memory."""
        if not states:
            return
        
        for state in states:
            if state.regime not in self._state:
                self._state[state.regime] = []
            
            # Update or append
            existing = None
            for i, s in enumerate(self._state[state.regime]):
                if s.engine_name == state.engine_name:
                    existing = i
                    break
            
            if existing is not None:
                self._state[state.regime][existing] = state
            else:
                self._state[state.regime].append(state)
        
        logger.debug(
            "inmemory_plasticity_saved",
            extra={"regimes": len(self._state), "total_states": len(states)},
        )
    
    def list_stored_regimes(self) -> List[str]:
        """Return list of all regimes with stored state."""
        return list(self._state.keys())
    
    def clear(self) -> None:
        """Clear all stored state."""
        self._state.clear()
        logger.info("inmemory_plasticity_cleared")
    
    def export_all(self) -> Dict[str, List[Dict]]:
        """Export all state as serializable dict.
        
        Useful for debugging and testing.
        """
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
