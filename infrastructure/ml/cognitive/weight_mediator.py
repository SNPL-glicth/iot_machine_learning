from __future__ import annotations

from typing import Dict, List

from .analysis.types import InhibitionState


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
        """Apply damped coupling between plasticity and inhibition.
        
        Args:
            plasticity_weights: Weights from PlasticityTracker.
            inhibition_states: Inhibition states from InhibitionGate.
        
        Returns:
            Mediated weights dict, normalized to sum to 1.0.
        """
        mediated = {}

        for state in inhibition_states:
            name = state.engine_name
            plastic_w = plasticity_weights.get(name, 0.0)
            inhibited_w = state.inhibited_weight
            base_w = state.base_weight

            prev_inh = self._last_inhibitions.get(name, 0.0)

            # If storage-MAE already penalized this engine (plastic_w dropped
            # more than 40 % below its base weight), reduce coupling so we do
            # not apply a second penalization on top.
            already_penalized = (base_w > 1e-9) and (plastic_w < base_w * 0.6)
            effective_coupling = 0.2 if already_penalized else self._coupling

            damping = 1.0 - effective_coupling * prev_inh

            mediated[name] = plastic_w * damping + inhibited_w * (1.0 - damping)
            self._last_inhibitions[name] = state.suppression_factor
        
        total = sum(mediated.values())
        if total < 1e-12:
            n = len(mediated)
            if n > 0:
                uniform = 1.0 / n
                return {k: uniform for k in mediated.keys()}
            return {}
        
        return {k: v / total for k, v in mediated.items()}
