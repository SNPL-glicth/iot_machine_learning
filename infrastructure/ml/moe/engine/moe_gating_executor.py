"""MoE Gating Executor: Handles gating, expert dispatch, and shadow gating."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..feature_context import FeatureContext
from ..registry import ExpertRegistry
from ..gating.strategy import GatingStrategy
from ..gating.tree_gating import TreeGatingNetwork
from ..gateway.expert_dispatcher import ExpertDispatcher
from ..config.moe_config import DISPATCH_TIMEOUT_MS


class MoEGatingExecutor:
    """Executes gating logic and dispatches to selected experts."""
    
    def __init__(
        self,
        registry: ExpertRegistry,
        gating: Optional[GatingStrategy] = None,
        shadow_gating: Optional[TreeGatingNetwork] = None,
    ):
        self._registry = registry
        self._gating = gating or ContextualRegimeGating(expert_ids=registry.list_all())
        self._shadow_gating = shadow_gating
        self._dispatcher = ExpertDispatcher(registry, timeout_ms=DISPATCH_TIMEOUT_MS)
    
    def execute_gating(
        self,
        feature_context: FeatureContext,
        sparsity_k: int,
    ) -> tuple[List[str], Dict[str, Any]]:
        """Run gating to select top-k experts. Returns (selected_experts, shadow_metadata)."""
        gating_result = self._gating.route(feature_context)
        selected_experts = gating_result.get_top_k(sparsity_k)
        
        shadow_metadata: Dict[str, Any] = {}
        if self._shadow_gating is not None:
            try:
                from iot_machine_learning.domain.model.context_vector import ContextVector
                shadow_ctx = ContextVector(
                    regime=feature_context.regime,
                    domain="iot",
                    n_points=len(feature_context.signal_features) if hasattr(feature_context, 'signal_features') else 0,
                    signal_features={
                        "mean": feature_context.mean,
                        "std": feature_context.std,
                        "slope": feature_context.slope,
                    },
                )
                shadow_probs = self._shadow_gating.route(shadow_ctx)
                max_diff = 0.0
                for eid in set(gating_result.probabilities.keys()) | set(shadow_probs.probabilities.keys()):
                    p1 = gating_result.probabilities.get(eid, 0.0)
                    p2 = shadow_probs.probabilities.get(eid, 0.0)
                    max_diff = max(max_diff, abs(p1 - p2))
                shadow_metadata = {
                    "shadow_gating": {
                        "top_expert": shadow_probs.top_expert,
                        "max_prob_diff": round(max_diff, 4),
                        "shadow_entropy": round(shadow_probs.entropy, 4),
                    }
                }
            except Exception as exc:
                import logging
                logging.getLogger("moe.shadow").warning(
                    "shadow_gating_failed", extra={"error": str(exc)}
                )
                shadow_metadata = {"shadow_gating": {"error": str(exc)}}
        
        return selected_experts, shadow_metadata
    
    def dispatch_experts(self, expert_ids: List[str], window) -> Dict[str, Any]:
        """Dispatch to selected experts and return outputs."""
        return self._dispatcher.dispatch(expert_ids, window)
    
    def get_regime(self, feature_context: FeatureContext) -> str:
        """Extract regime from feature context."""
        return feature_context.regime