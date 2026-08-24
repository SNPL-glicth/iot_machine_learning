"""ContextualRegimeGating — routing based on numeric features from pipeline."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .base import GatingProbs
from ..feature_context import FeatureContext
from .regime_weight_loader import load_regime_weights, load_equipment_weights
from .performance_tracker import ExpertPerformanceTracker
from .feature_adjuster import FeatureAdjuster


class ContextualRegimeGating:
    """Gating that uses numeric features from pipeline for routing."""
    
    DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = load_regime_weights()
    EQUIPMENT_REGIME_WEIGHTS: Dict[str, Dict[str, Dict[str, float]]] = load_equipment_weights()
    
    def __init__(
        self,
        regime_weights: Optional[Dict[str, Dict[str, float]]] = None,
        expert_ids: Optional[List[str]] = None,
        noise_boost_factor: float = 1.5,
        slope_threshold: float = 0.01,
        stability_bonus: float = 0.05,
    ) -> None:
        self._regime_weights = regime_weights or dict(self.DEFAULT_WEIGHTS)
        self._expert_ids = expert_ids or []
        self._slope_threshold = slope_threshold
        self._stability_bonus = stability_bonus
        self._performance_tracker = ExpertPerformanceTracker()
        self._feature_adjuster = FeatureAdjuster(slope_threshold=slope_threshold)
        self._last_top_expert: Optional[str] = None
    
    def route(self, feature_context: FeatureContext) -> GatingProbs:
        """Decide probability distribution using numeric features."""
        regime = feature_context.regime
        equipment_class = getattr(feature_context, "equipment_class", "GENERIC")
        
        equipment_override = self.EQUIPMENT_REGIME_WEIGHTS.get(equipment_class, {})
        base_weights = equipment_override.get(regime) or self._regime_weights.get(regime, {})
        
        weights: Dict[str, float] = {
            eid: base_weights.get(eid, 0.0) for eid in self._expert_ids
        }
        
        # Apply feature-based adjustments
        weights = self._feature_adjuster.adjust_by_std(weights, feature_context.std)
        weights = self._feature_adjuster.adjust_by_slope(weights, feature_context.slope)
        weights = self._feature_adjuster.adjust_by_noise(weights, feature_context.noise_ratio)
        weights = self._feature_adjuster.adjust_by_curvature(weights, feature_context.curvature)
        weights = self._adjust_by_performance(weights)
        
        # Temporal stability: slight bonus to last top_expert
        if self._last_top_expert is not None and self._last_top_expert in weights:
            weights[self._last_top_expert] *= (1.0 + self._stability_bonus)
        
        probabilities = self._normalize(weights)
        entropy = self._compute_entropy(probabilities)
        
        top_expert = max(probabilities.items(), key=lambda x: x[1])[0] if probabilities else ""
        self._last_top_expert = top_expert
        
        return GatingProbs(
            probabilities=probabilities,
            entropy=entropy,
            top_expert=top_expert,
            metadata={
                "regime": regime,
                "std": feature_context.std,
                "slope": feature_context.slope,
                "noise_ratio": feature_context.noise_ratio,
                "curvature": feature_context.curvature,
                "raw_weights": weights,
                "equipment_class": equipment_class,
                "used_equipment_override": equipment_class in self.EQUIPMENT_REGIME_WEIGHTS,
                "performance_scores": {
                    eid: round(self._performance_tracker.get_reliability(eid), 3)
                    for eid in self._expert_ids
                },
            },
        )
    
    def explain(
        self, feature_context: FeatureContext, probs: GatingProbs
    ) -> str:
        """Explain routing decision in human language."""
        regime = feature_context.regime
        top = probs.top_expert
        top_prob = probs.max_probability
        
        reasons = [f"régimen={regime}"]
        
        if feature_context.std > 1.0:
            reasons.append(f"std={feature_context.std:.2f}")
        if abs(feature_context.slope) > self._slope_threshold:
            reasons.append(f"slope={feature_context.slope:.4f}")
        if feature_context.noise_ratio > 0.3:
            reasons.append(f"noise={feature_context.noise_ratio:.2f}")
        if abs(feature_context.curvature) > 0.001:
            reasons.append(f"curvature={feature_context.curvature:.4f}")
        
        perf_scores = probs.metadata.get("performance_scores", {})
        if perf_scores:
            top_perf = max(perf_scores.items(), key=lambda x: x[1])
            reasons.append(f"best_history={top_perf[0]}({top_perf[1]:.2f})")
        
        return (
            f"ContextualRegimeGating: top_expert={top}({top_prob:.2f}) "
            f"entropy={probs.entropy:.3f} | {' | '.join(reasons)}"
        )
    
    def get_expert_ids(self) -> List[str]:
        return list(self._expert_ids)
    
    def _adjust_by_performance(self, weights: Dict[str, float]) -> Dict[str, float]:
        adjusted = dict(weights)
        for eid in adjusted:
            rel = self._performance_tracker.get_reliability(eid)
            if rel < 0.3:
                adjusted[eid] *= 0.5
            elif rel > 0.7:
                adjusted[eid] *= (1.0 + (rel - 0.7) * 0.3)
        return adjusted
    
    @staticmethod
    def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total < 1e-9:
            n = len(weights)
            return {k: 1.0 / n for k in weights} if n > 0 else {}
        return {k: v / total for k, v in weights.items()}
    
    @staticmethod
    def _compute_entropy(probabilities: Dict[str, float]) -> float:
        entropy = 0.0
        for p in probabilities.values():
            if p > 1e-9:
                entropy -= p * math.log2(p)
        return entropy