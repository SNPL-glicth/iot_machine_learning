"""Weighted fusion and engine ranking for the cognitive orchestrator.

Combines inhibited weights with perceptions to produce a fused
prediction and identify the primary (highest-weight) engine.

Fusion formula:
    final_prediction = Σ (prediction_i × w_i) / Σ w_i
    final_confidence = Σ (confidence_i × w_i) / Σ w_i
    final_trend      = majority vote weighted by w_i

Pure logic — no I/O, no state, no logging.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from ..analysis.types import EnginePerception, InhibitionState


class WeightedFusion:
    """Fuses multiple engine perceptions into a single prediction."""

    def fuse(
        self,
        perceptions: List[EnginePerception],
        inhibition_states: List[InhibitionState],
        neighbor_trends: Optional[Dict[str, str]] = None,
        signal_std: float = 0.0,
    ) -> Tuple[float, float, str, Dict[str, float], str, str]:
        """Compute weighted fusion of engine perceptions.

        Args:
            perceptions: Each engine's perception.
            inhibition_states: Post-inhibition weights.
            neighbor_trends: Optional dict of {series_id: "up"/"down"/"stable"}
                from correlated neighbours. When provided, a small spatial bias
                (10% of signal_std) is applied to the fused value.
            signal_std: Standard deviation of the current signal window,
                used to scale the neighbour bias. Defaults to 0 (no bias).

        Returns:
            Tuple of:
                - fused_value: Weighted average prediction.
                - fused_confidence: Weighted average confidence.
                - fused_trend: Majority-vote trend.
                - final_weights: Normalized weights used.
                - selected_engine: Engine with highest weight.
                - selection_reason: Why that engine was selected.
        """
        if not perceptions:
            return 0.0, 0.0, "stable", {}, "none", "no_engines"

        # Build weight map from inhibition states
        weight_map: Dict[str, float] = {
            s.engine_name: s.inhibited_weight
            for s in inhibition_states
        }

        # Normalize weights
        total_w = sum(weight_map.values())
        if total_w < 1e-12:
            n = len(perceptions)
            weight_map = {p.engine_name: 1.0 / n for p in perceptions}
            total_w = 1.0

        norm_weights = {k: v / total_w for k, v in weight_map.items()}

        # Weighted average of predictions and confidence
        fused_value = 0.0
        fused_confidence = 0.0
        trend_votes: Dict[str, float] = {"up": 0.0, "down": 0.0, "stable": 0.0}

        for p in perceptions:
            w = norm_weights.get(p.engine_name, 0.0)
            fused_value += p.predicted_value * w
            fused_confidence += p.confidence * w
            trend_votes[p.trend] += w

        # Apply spatial bias from correlated neighbours
        if neighbor_trends and signal_std > 1e-9:
            up_count = sum(1 for t in neighbor_trends.values() if t == "up")
            down_count = sum(1 for t in neighbor_trends.values() if t == "down")
            bias_factor = (up_count - down_count) / max(len(neighbor_trends), 1)
            fused_value += bias_factor * signal_std * 0.1

        # Majority vote for trend
        fused_trend = max(trend_votes, key=trend_votes.get)  # type: ignore[arg-type]

        # Identify primary engine (highest weight)
        selected = max(norm_weights, key=norm_weights.get)  # type: ignore[arg-type]
        reason = self._build_reason(selected, norm_weights, inhibition_states)

        return (
            fused_value,
            fused_confidence,
            fused_trend,
            norm_weights,
            selected,
            reason,
        )

    def _build_reason(
        self,
        selected: str,
        weights: Dict[str, float],
        states: List[InhibitionState],
    ) -> str:
        """Build human-readable selection reason."""
        w = weights.get(selected, 0.0)
        parts = [f"highest_weight={w:.3f}"]

        # Check if any engines were inhibited
        inhibited = [
            s for s in states
            if s.suppression_factor > 0.01
        ]
        if inhibited:
            names = [s.engine_name for s in inhibited]
            parts.append(f"inhibited=[{','.join(names)}]")

        # Check if it's a single-engine scenario
        active = [k for k, v in weights.items() if v > 0.01]
        if len(active) == 1:
            parts.append("single_active_engine")

        return "; ".join(parts)
