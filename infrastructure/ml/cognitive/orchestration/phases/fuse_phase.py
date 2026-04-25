"""Fuse Phase — MED-1 Refactoring.

Weighted fusion and spatial correction.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Tuple

from ...fusion import hampel_filter

if TYPE_CHECKING:
    from . import PipelineContext
    from ...analysis.types import EnginePerception, InhibitionState

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid_env_float", extra={"name": name, "value": raw})
        return default


# IMP-2 kill switches.
ML_HAMPEL_ENABLED: bool = _env_bool("ML_HAMPEL_ENABLED", True)
ML_HAMPEL_K: float = _env_float("ML_HAMPEL_K", 3.0)


def _compute_robust_gradient(values: list) -> float:
    """Compute OLS gradient over values."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if abs(den) > 1e-9 else 0.0


def _apply_spatial_correction(
    base_prediction: float,
    neighbors: list,
    neighbor_values: dict,
    max_correction_pct: float = 0.15,
    min_gradient_samples: int = 3,
) -> float:
    """Apply spatial gradient correction from correlated neighbors."""
    if not neighbors:
        return base_prediction
    
    valid_neighbors = []
    for neighbor_id, correlation in neighbors:
        if abs(correlation) > 0.5 and neighbor_id in neighbor_values:
            values = neighbor_values[neighbor_id]
            if len(values) >= min_gradient_samples:
                gradient = _compute_robust_gradient(values)
                valid_neighbors.append((neighbor_id, correlation, gradient))
    
    if not valid_neighbors:
        return base_prediction
    
    weighted_gradient = 0.0
    total_abs_correlation = 0.0
    
    for neighbor_id, correlation, gradient in valid_neighbors:
        weighted_gradient += correlation * gradient
        total_abs_correlation += abs(correlation)
    
    if total_abs_correlation < 1e-9:
        return base_prediction
    
    gradient = weighted_gradient / total_abs_correlation
    correction = gradient * (total_abs_correlation / len(valid_neighbors))
    
    max_correction = abs(base_prediction) * max_correction_pct
    correction = max(-max_correction, min(max_correction, correction))
    
    return base_prediction + correction


class FusePhase:
    """Phase 5: Weighted fusion and spatial correction."""
    
    @property
    def name(self) -> str:
        return "fuse"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute fusion phase."""
        orchestrator = ctx.orchestrator
        
        # IMP-2: pre-fusion Hampel outlier filter.
        filtered_perceptions, filtered_states, fusion_flags, hampel_diag = (
            self._apply_hampel(ctx.perceptions, ctx.inhibition_states)
        )
        
        # Perform fusion
        fused_result = orchestrator._fusion.fuse(
            filtered_perceptions,
            filtered_states,
            neighbor_trends=ctx.neighbor_trends,
            signal_std=ctx.profile.std if ctx.profile else 0.0,
        )
        
        fused_val, fused_conf, fused_trend, final_weights, selected, reason = fused_result
        
        # Apply spatial correction
        fused_val = _apply_spatial_correction(
            fused_val, ctx.neighbors, ctx.neighbor_values
        )
        
        # Field smoothing via correlation port
        if orchestrator._correlation_port and ctx.neighbors:
            fused_val = self._apply_field_smoothing(ctx, fused_val)
        
        # Determine fusion method (from filtered perceptions, post-Hampel)
        method = "weighted_average" if len(filtered_perceptions) > 1 else "single_engine"
        
        # Update explanation builder
        if ctx.explanation and hasattr(ctx.explanation, 'set_fusion'):
            ctx.explanation.set_fusion(
                fused_val, fused_conf, fused_trend,
                final_weights, selected, reason, method,
            )
        
        return ctx.with_field(
            fused_value=fused_val,
            fused_confidence=fused_conf,
            fused_trend=fused_trend,
            final_weights=final_weights,
            selected_engine=selected,
            selection_reason=reason,
            fusion_method=method,
            fusion_flags=fusion_flags,
            hampel_diagnostic=hampel_diag,
        )
    
    def _apply_hampel(
        self,
        perceptions: List["EnginePerception"],
        inhibition_states: List["InhibitionState"],
    ) -> Tuple[
        List["EnginePerception"],
        List["InhibitionState"],
        List[str],
        dict,
    ]:
        """Apply Hampel filter before weighted fusion.
        
        Returns (filtered_perceptions, filtered_states, fusion_flags, diagnostic_dict).
        Falls back to raw perceptions when:
            * kill switch disabled (ML_HAMPEL_ENABLED=0);
            * fewer than 3 perceptions (Hampel no-ops by design);
            * all perceptions would be rejected (defensive).
        """
        flags: List[str] = []
        if not ML_HAMPEL_ENABLED or not perceptions:
            return list(perceptions), list(inhibition_states), flags, {}
        
        result = hampel_filter(perceptions, k=ML_HAMPEL_K)
        if not result.kept:
            flags.append("hampel_all_rejected_bypassed")
            logger.warning(
                "hampel_all_rejected",
                extra={"n_perceptions": len(perceptions), "median": result.median},
            )
            return list(perceptions), list(inhibition_states), flags, result.to_dict()
        
        if result.rejected:
            flags.append(f"hampel_rejected:{len(result.rejected)}")
            logger.info(
                "hampel_outliers_rejected",
                extra={
                    "n_rejected": len(result.rejected),
                    "median": round(result.median, 4),
                    "mad": round(result.mad, 4),
                    "rejected_engines": [r[0] for r in result.rejected],
                },
            )
        
        kept_names = {p.engine_name for p in result.kept}
        filtered_states = [s for s in inhibition_states if s.engine_name in kept_names]
        return result.kept, filtered_states, flags, result.to_dict()
    
    def _apply_field_smoothing(self, ctx: PipelineContext, fused_val: float) -> float:
        """Apply spatial smoothing with high-correlation neighbors."""
        orchestrator = ctx.orchestrator
        
        high_corr_neighbors = [
            (nid, corr) for nid, corr in ctx.neighbors 
            if abs(corr) > 0.7
        ]
        
        if not high_corr_neighbors:
            return fused_val
        
        try:
            predictions_to_smooth = {ctx.series_id: fused_val}
            
            for neighbor_id, _ in high_corr_neighbors:
                if orchestrator._storage:
                    neighbor_pred = orchestrator._storage.get_latest_prediction_for_series(
                        neighbor_id
                    )
                    if neighbor_pred and hasattr(neighbor_pred, 'predicted_value'):
                        predictions_to_smooth[neighbor_id] = neighbor_pred.predicted_value
            
            if len(predictions_to_smooth) >= 2:
                smoothed = orchestrator._correlation_port.smooth_with_field(
                    predictions_to_smooth,
                    smoothing_factor=0.2,
                )
                if smoothed and ctx.series_id in smoothed:
                    original_val = fused_val
                    fused_val = smoothed[ctx.series_id]
                    logger.debug("field_smoothing_applied", extra={
                        "original": round(original_val, 4),
                        "smoothed": round(fused_val, 4),
                        "n_neighbors": len(predictions_to_smooth) - 1,
                    })
        except Exception as e:
            logger.debug(f"field_smoothing_failed: {e}")
        
        return fused_val
