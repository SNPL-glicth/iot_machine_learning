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

import json
import logging
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from core.parameters.numerical_constants import EPSILON
from ..analysis.types import EnginePerception, InhibitionState
from iot_machine_learning.core.ensemble.ensemble_watchdog import EnsembleWatchdog
from iot_machine_learning.core.ensemble.forced_recovery import ForcedRecoveryManager

logger = logging.getLogger(__name__)


class WeightedFusion:
    """Fuses multiple engine perceptions into a single prediction."""

    def __init__(
        self,
        correlation_analyzer: Optional["EngineCorrelationAnalyzer"] = None,
        decorrelator: Optional["EnsembleDecorrelator"] = None,
        watchdog: Optional[EnsembleWatchdog] = None,
        recovery_manager: Optional[ForcedRecoveryManager] = None,
    ) -> None:
        self._correlation_analyzer = correlation_analyzer
        self._decorrelator = decorrelator
        self._watchdog = watchdog
        self._recovery_manager = recovery_manager

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

        # Correlation analysis and decorrelation (optional)
        if (
            self._correlation_analyzer is not None
            and self._decorrelator is not None
            and len(perceptions) > 2
        ):
            # Collect predictions for correlation analysis
            predictions_dict = {
                p.engine_name: np.array([p.predicted_value]) for p in perceptions
            }
            correlation_result = self._correlation_analyzer.analyze(predictions_dict)

            if correlation_result.max_correlation > 0.7:
                weight_map, _ = self._decorrelator.apply_if_needed(
                    weight_map, correlation_result
                )
                logger.info(
                    "ensemble_decorrelation_applied",
                    extra={
                        "max_correlation": round(correlation_result.max_correlation, 3),
                        "diversity_score": round(
                            self._decorrelator.compute_diversity_score(
                                correlation_result
                            ),
                            3,
                        ),
                    },
                )

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
            confidence = max(0.0, min(1.0, p.confidence))
            if not (0.0 <= p.confidence <= 1.0):
                logger.warning(json.dumps({
                    "event": "confidence_out_of_range",
                    "component": "WeightedFusion",
                    "engine": p.engine_name,
                    "raw_confidence": p.confidence,
                    "clamped_to": confidence,
                }))
            fused_confidence += confidence * w
            trend_votes[p.trend] += w

        # Apply spatial bias from correlated neighbours
        if neighbor_trends and signal_std > EPSILON.DIVISION:
            up_count = sum(1 for t in neighbor_trends.values() if t == "up")
            down_count = sum(1 for t in neighbor_trends.values() if t == "down")
            bias_factor = (up_count - down_count) / max(len(neighbor_trends), 1)
            fused_value += bias_factor * signal_std * 0.1

        # Majority vote for trend
        fused_trend = max(trend_votes, key=trend_votes.get)  # type: ignore[arg-type]

        # Identify primary engine (highest weight)
        selected = max(norm_weights, key=norm_weights.get)  # type: ignore[arg-type]
        reason = self._build_reason(selected, norm_weights, inhibition_states)

        # POST-FUSION: evaluate ensemble health and trigger recovery if needed
        if self._watchdog:
            suppressions = {s.engine_name: s.suppression_factor for s in inhibition_states}
            snapshot = self._watchdog.evaluate(weights=norm_weights, suppressions=suppressions)
            
            if self._watchdog.should_trigger_recovery(snapshot) and self._recovery_manager:
                # Get engine errors if available (from perception metadata)
                engine_errors = {}
                for p in perceptions:
                    if hasattr(p, 'metadata') and p.metadata:
                        error = p.metadata.get('recent_error')
                        if error is not None:
                            engine_errors[p.engine_name] = error
                
                recovery_result = self._recovery_manager.execute(
                    snapshot=snapshot,
                    current_weights=norm_weights,
                    current_suppressions=suppressions,
                    engine_errors=engine_errors if engine_errors else None,
                )
                
                if recovery_result.success and recovery_result.weight_adjustments:
                    # Apply recovery adjustments to weights
                    for engine, new_weight in recovery_result.weight_adjustments.items():
                        norm_weights[engine] = new_weight
                    
                    # Re-normalize weights
                    total_w = sum(norm_weights.values())
                    if total_w > 1e-12:
                        norm_weights = {k: v / total_w for k, v in norm_weights.items()}
                    
                    # Re-compute fusion with recovered weights
                    fused_value = 0.0
                    fused_confidence = 0.0
                    trend_votes: Dict[str, float] = {"up": 0.0, "down": 0.0, "stable": 0.0}
                    
                    for p in perceptions:
                        w = norm_weights.get(p.engine_name, 0.0)
                        fused_value += p.predicted_value * w
                        confidence = max(0.0, min(1.0, p.confidence))
                        fused_confidence += confidence * w
                        trend_votes[p.trend] += w
                    
                    # Re-apply spatial bias
                    if neighbor_trends and signal_std > EPSILON.DIVISION:
                        up_count = sum(1 for t in neighbor_trends.values() if t == "up")
                        down_count = sum(1 for t in neighbor_trends.values() if t == "down")
                        bias_factor = (up_count - down_count) / max(len(neighbor_trends), 1)
                        fused_value += bias_factor * signal_std * 0.1
                    
                    fused_trend = max(trend_votes, key=trend_votes.get)  # type: ignore[arg-type]
                    selected = max(norm_weights, key=norm_weights.get)  # type: ignore[arg-type]
                    reason = self._build_reason(selected, norm_weights, inhibition_states)
                    
                    logger.info(
                        "ensemble_recovery_applied",
                        extra={
                            "strategy": recovery_result.strategy_used.value,
                            "engines_recovered": recovery_result.engines_recovered,
                            "snapshot_health": snapshot.health.value,
                            "recovery_reason": recovery_result.reason,
                        }
                    )

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
