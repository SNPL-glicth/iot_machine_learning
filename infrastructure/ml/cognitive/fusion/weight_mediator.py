from __future__ import annotations

from typing import Dict, List

from ..analysis.types import InhibitionState


class ForceBalanceService:
    """Physics-based force balance for weight mediation.
    
    Models plasticity as attractive force and inhibition as repulsive force.
    Computes equilibrium weight using: w_eq = (F_plastic - F_inhibition) / (mass + damping)
    """
    
    @staticmethod
    def compute_equilibrium_weight(
        plastic_force: float,
        inhibition_force: float,
        mass: float = 1.0,
        damping: float = 0.3,
    ) -> float:
        """Compute equilibrium weight from force balance.
        
        At equilibrium: F_plastic - F_inhibition = m·a + b·v
        When a=0, v=0: w_eq = net_force / (mass + damping)
        
        Args:
            plastic_force: Attraction force (proportional to historical accuracy)
            inhibition_force: Repulsion force (proportional to current instability)
            mass: Engine "mass" (resistance to change)
            damping: Viscous damping coefficient
        
        Returns:
            Equilibrium weight in [0.0, 1.0]
        """
        net_force = plastic_force - inhibition_force
        equilibrium_displacement = net_force / (mass + damping)
        return max(0.0, min(1.0, equilibrium_displacement))


class WeightMediator:
    """Mediates between plasticity attraction and inhibition repulsion.
    
    Applies damped coupling to prevent oscillations between PlasticityTracker
    (which increases weights based on historical performance) and InhibitionGate
    (which suppresses weights based on current instability).
    """

    def __init__(self, coupling_strength: float = 0.5) -> None:
        self._coupling = coupling_strength
        self._last_inhibitions: Dict[str, float] = {}

    def mediate(
        self,
        plasticity_weights: Dict[str, float],
        inhibition_states: List[InhibitionState],
    ) -> Dict[str, float]:
        """Apply force balance between plasticity and inhibition.
        
        Args:
            plasticity_weights: Weights from PlasticityTracker.
            inhibition_states: Inhibition states from InhibitionGate.
        
        Returns:
            Mediated weights dict, normalized to sum to 1.0.
        """
        mediated = {}
        force_balance = ForceBalanceService()

        for state in inhibition_states:
            name = state.engine_name
            plastic_w = plasticity_weights.get(name, 0.0)
            inhibited_w = state.inhibited_weight
            
            plastic_force = plastic_w
            inhibition_force = state.suppression_factor
            
            mediated[name] = force_balance.compute_equilibrium_weight(
                plastic_force=plastic_force,
                inhibition_force=inhibition_force,
                mass=1.0,
                damping=0.3,
            )
            self._last_inhibitions[name] = state.suppression_factor
        
        total = sum(mediated.values())
        if total < 1e-12:
            n = len(mediated)
            if n > 0:
                uniform = 1.0 / n
                return {k: uniform for k in mediated.keys()}
            return {}
        
        return {k: v / total for k, v in mediated.items()}
