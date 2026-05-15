"""Fuse Phase — MED-1 Refactoring.

Weighted fusion and spatial correction.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

from core.parameters.numerical_constants import EPSILON

from ...fusion import hampel_filter
from .fuse_phase_config import FusePhaseConfig
from .hampel_fallback_strategy import (
    HampelFallbackStrategy,
    MedianClosestFallbackStrategy,
)

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


# Factory helper to build config from env vars
def _build_fuse_phase_config_from_env() -> FusePhaseConfig:
    """Build FusePhaseConfig from environment variables.
    
    Separates env var reading from phase logic (DIP).
    """
    return FusePhaseConfig(
        hampel_k=_env_float("ML_HAMPEL_K", 3.0),
        hampel_enabled=_env_bool("ML_HAMPEL_ENABLED", True),
        # Other parameters use dataclass defaults
    )


def _compute_robust_gradient(values: list) -> float:
    """Compute OLS gradient over values."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if abs(den) > EPSILON.DIVISION else 0.0


def _validate_correlation_quality(
    correlation: float,
    n_samples: int,
    config: FusePhaseConfig,
) -> bool:
    """Validate correlation is statistically significant (COG-SEV-2).
    
    Args:
        correlation: Correlation coefficient.
        n_samples: Number of samples used to compute correlation.
        config: FusePhaseConfig with min_samples and p_value threshold.
    
    Returns:
        True if correlation is statistically significant.
    
    Applies COG-SEV-2: Filters out spurious correlations using t-test.
    Uses Student's t-distribution table for accurate critical values.
    """
    if n_samples < config.min_samples_for_significance:
        return False
    
    # Compute t-statistic for correlation
    # t = r * sqrt(n-2) / sqrt(1 - r^2)
    r = abs(correlation)
    if r >= 0.9999:  # Perfect correlation
        return True
    
    t_stat = r * ((n_samples - 2) ** 0.5) / ((1 - r * r) ** 0.5)
    
    # Get critical t-value from config (uses accurate t-distribution table)
    df = n_samples - 2
    t_critical = config.get_t_critical(df)
    
    return t_stat > t_critical


def _apply_spatial_correction(
    base_prediction: float,
    neighbors: list,
    neighbor_values: dict,
    config: FusePhaseConfig,
) -> float:
    """Apply spatial gradient correction from correlated neighbors.
    
    Args:
        base_prediction: Base prediction value.
        neighbors: List of (neighbor_id, correlation) tuples.
        neighbor_values: Dict of neighbor_id -> historical values.
        config: FusePhaseConfig with spatial correction parameters.
    
    Returns:
        Corrected prediction value.
    
    COG-SEV-2: Filters neighbors to only those with:
    - abs(correlation) > config.min_correlation
    - Statistically significant correlation (p < config.p_value_threshold)
    """
    if not neighbors:
        return base_prediction
    
    valid_neighbors = []
    for neighbor_id, correlation in neighbors:
        # Filter by correlation threshold
        if abs(correlation) > config.min_correlation and neighbor_id in neighbor_values:
            values = neighbor_values[neighbor_id]
            if len(values) >= config.min_gradient_samples:
                # Validate correlation quality
                if _validate_correlation_quality(correlation, len(values), config):
                    gradient = _compute_robust_gradient(values)
                    valid_neighbors.append((neighbor_id, correlation, gradient))
    
    if not valid_neighbors:
        return base_prediction
    
    weighted_gradient = 0.0
    total_abs_correlation = 0.0
    
    for neighbor_id, correlation, gradient in valid_neighbors:
        weighted_gradient += correlation * gradient
        total_abs_correlation += abs(correlation)
    
    if total_abs_correlation < EPSILON.CORRELATION:
        return base_prediction
    
    gradient = weighted_gradient / total_abs_correlation
    correction = gradient * (total_abs_correlation / len(valid_neighbors))
    
    max_correction = abs(base_prediction) * config.max_correction_pct
    correction = max(-max_correction, min(max_correction, correction))
    
    return base_prediction + correction


class FusePhase:
    """Phase 5: Weighted fusion and spatial correction.
    
    Applies DIP: Depends on FusePhaseConfig abstraction, not env vars.
    Applies SRP: Configuration is injected, not read internally.
    Applies OCP: New parameters only require extending FusePhaseConfig.
    """
    
    def __init__(
        self,
        config: FusePhaseConfig = None,
        hampel_fallback_strategy: HampelFallbackStrategy = None,
        temperature_scaler = None,
    ) -> None:
        """Initialize fuse phase with injectable configuration.

        Args:
            config: FusePhaseConfig with all parameters. Defaults to standard config.
            hampel_fallback_strategy: Strategy for Hampel all-rejected case.
                Defaults to MedianClosestFallbackStrategy.

        Applies DIP: Dependencies are injected, not constructed internally.
        """
        self._config = config or FusePhaseConfig()
        self._hampel_fallback_strategy = (
            hampel_fallback_strategy or MedianClosestFallbackStrategy()
        )
        self._temperature_scaler = temperature_scaler  # FASE-9: Optional temperature scaling
    
    @property
    def name(self) -> str:
        return "fuse"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute fusion phase."""
        orchestrator = ctx.orchestrator
        
        # IMP-2: pre-fusion Hampel outlier filter.
        filtered_perceptions, filtered_states, fusion_flags, hampel_diag = (
            self._apply_hampel_filter(ctx.perceptions, ctx.inhibition_states)
        )
        
        # Perform fusion
        fused_result = orchestrator._fusion.fuse(
            filtered_perceptions,
            filtered_states,
            neighbor_trends=ctx.neighbor_trends,
            signal_std=ctx.profile.std if ctx.profile else 0.0,
        )
        
        fused_val, fused_conf, fused_trend, final_weights, selected, reason = fused_result
        
        # FASE-9: Apply temperature scaling to confidence if scaler available
        if self._temperature_scaler and ctx.profile:
            regime = ctx.profile.regime.value if hasattr(ctx.profile.regime, 'value') else str(ctx.profile.regime)
            scaling_result = self._temperature_scaler.scale(fused_conf, regime=regime)
            fused_conf = scaling_result.scaled_confidence
            logger.debug(
                "temperature_scaling_applied",
                extra={
                    "original_conf": round(fused_conf, 4),
                    "scaled_conf": round(scaling_result.scaled_confidence, 4),
                    "regime": regime,
                    "temperature": round(scaling_result.temperature, 4),
                },
            )
        
        # Apply spatial correction
        fused_val = _apply_spatial_correction(
            fused_val, ctx.neighbors, ctx.neighbor_values, self._config
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
    
    def _apply_hampel_filter(
        self,
        perceptions: List["EnginePerception"],
        inhibition_states: List["InhibitionState"],
    ) -> Tuple[List["EnginePerception"], List["InhibitionState"], List[str], dict]:
        """Apply Hampel filter to reject outlier predictions.
        
        Returns:
            * filtered perceptions
            * filtered inhibition states
            * flags (e.g. "hampel_all_rejected_using_median")
            * diagnostic dict
        
        COG-CRIT-2: When all rejected, uses fallback strategy (default: median-closest)
        instead of bypassing filter.
        """
        flags: List[str] = []
        if not self._config.hampel_enabled or not perceptions:
            return list(perceptions), list(inhibition_states or []), flags, {}
        
        result = hampel_filter(perceptions, k=self._config.hampel_k)
        if not result.kept:
            # COG-CRIT-2: Use fallback strategy instead of bypass
            states = inhibition_states or []
            selected_perceptions, selected_states, reason = (
                self._hampel_fallback_strategy.select_fallback(
                    perceptions, states, result.median
                )
            )
            flags.append(reason)
            logger.warning(
                "hampel_all_rejected_fallback",
                extra={
                    "n_perceptions": len(perceptions),
                    "median": result.median,
                    "fallback_strategy": self._hampel_fallback_strategy.__class__.__name__,
                    "selected_engine": selected_perceptions[0].engine_name if selected_perceptions else "none",
                },
            )
            return selected_perceptions, selected_states, flags, result.to_dict()
        
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
        states = inhibition_states or []
        filtered_states = [s for s in states if s.engine_name in kept_names]
        return result.kept, filtered_states, flags, result.to_dict()
    
    def _apply_field_smoothing(self, ctx: PipelineContext, fused_val: float) -> float:
        """Apply spatial smoothing with high-correlation neighbors."""
        orchestrator = ctx.orchestrator
        
        high_corr_neighbors = [
            (nid, corr) for nid, corr in ctx.neighbors 
            if abs(corr) > self._config.min_correlation
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
            
            if len(predictions_to_smooth) >= self._config.field_smoothing_min_neighbors:
                smoothed = orchestrator._correlation_port.smooth_with_field(
                    predictions_to_smooth,
                    smoothing_factor=self._config.smoothing_factor,
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
