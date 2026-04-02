"""Fuse Phase — MED-1 Refactoring.

Weighted fusion and spatial correction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import PipelineContext

logger = logging.getLogger(__name__)


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
        
        # Perform fusion
        fused_result = orchestrator._fusion.fuse(
            ctx.perceptions,
            ctx.inhibition_states,
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
        
        # Determine fusion method
        method = "weighted_average" if len(ctx.perceptions) > 1 else "single_engine"
        
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
        )
    
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
