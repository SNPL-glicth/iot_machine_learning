"""Reason phase: inhibit + adapt + fuse."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

from iot_machine_learning.infrastructure.ml.cognitive.inhibition import InhibitionGate, InhibitionConfig
from iot_machine_learning.infrastructure.ml.cognitive.fusion import WeightedFusion

logger = logging.getLogger(__name__)

_PLASTICITY_AVAILABLE = True
try:
    from iot_machine_learning.infrastructure.ml.cognitive.plasticity import PlasticityTracker
except (ImportError, ModuleNotFoundError):
    _PLASTICITY_AVAILABLE = False


class ReasonPhase:
    """Phase 4: Inhibit + Adapt + Fuse (reuses existing cognitive components)."""

    def __init__(self, enable_plasticity: bool = True) -> None:
        self._inhibition = InhibitionGate(InhibitionConfig())
        self._fusion = WeightedFusion()
        
        self._plasticity = None
        if enable_plasticity and _PLASTICITY_AVAILABLE:
            self._plasticity = PlasticityTracker()

    def execute(
        self,
        perceptions: List,
        domain: str,
        series_id: str,
        timing: Dict[str, float],
    ) -> Tuple[float, float, str, Dict[str, float], str, str, str]:
        """Execute reason phase.

        Args:
            perceptions: List of EnginePerception objects
            domain: Classified domain
            series_id: Series identifier
            timing: Pipeline timing dict

        Returns:
            Tuple of (fused_val, fused_conf, fused_trend, final_weights, selected, reason, method)
        """
        t0 = time.monotonic()
        
        engine_names = [p.engine_name for p in perceptions]
        
        base_weights = {}
        if self._plasticity and self._plasticity.has_history(domain):
            base_weights = self._plasticity.get_weights(domain, engine_names)
        else:
            n = len(engine_names)
            base_weights = {name: 1.0 / n for name in engine_names}
        
        timing["adapt"] = (time.monotonic() - t0) * 1000
        
        t0 = time.monotonic()
        inh_states = self._inhibition.compute(
            perceptions, base_weights, series_id=series_id
        )
        timing["inhibit"] = (time.monotonic() - t0) * 1000
        
        t0 = time.monotonic()
        (fused_val, fused_conf, fused_trend,
         final_weights, selected, reason) = self._fusion.fuse(
            perceptions, inh_states
        )
        
        method = "weighted_average" if len(perceptions) > 1 else "single_engine"
        
        timing["fuse"] = (time.monotonic() - t0) * 1000
        
        return fused_val, fused_conf, fused_trend, final_weights, selected, reason, method
