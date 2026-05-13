"""Seasonal Decomposition Phase — removes seasonal component from signal.

Extracts seasonal patterns using STL or FFT, then subtracts seasonal component
from values before they reach PerceivePhase and VotingAnomalyDetector.

This prevents cyclic patterns from being flagged as anomalies.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional, Tuple, List

from ...seasonal import STLDecomposer, FFTSeasonalityDetector

if TYPE_CHECKING:
    from . import PipelineContext

logger = logging.getLogger(__name__)


class SeasonalDecompositionPhase:
    """Phase 2: Seasonal component removal.
    
    Decomposes time series into trend + seasonal + residual.
    Replaces ctx.values with residual (seasonal removed).
    
    Dependency injection:
    - Decomposer (STL or FFT) selected via flags
    - Period configuration from SeriesValuesStore or default
    
    HIDDEN ASSUMPTIONS (FASE-25):
    1. HOURLY DATA: seasonal_period_default=24 asume datos horarios
       (24h = 1 día). Para datos a 1-min: period=1440. Para datos
       diarios: period=7 (semanal).
       PENDING_CALIBRATION: Ajustar ML_SEASONAL_PERIOD_DEFAULT según
       frecuencia real. Ver ML_SAMPLING_FREQUENCY_HZ en cognitive_config.py.
    2. PERIODIC DATA: Seasonal decomposition falla silenciosamente
       en datos no-periódicos. Engine devuelve fallback en ese caso.
    3. MIN POINTS: seasonal_min_points=48 = 2 × period_default(24).
       Intencional: mínimo 2 ciclos completos para descomposición válida.
    """
    
    def __init__(
        self,
        enable_seasonality: bool = False,
        seasonal_period_default: int = 24,
        seasonal_use_stl: bool = False,
        seasonal_min_points: int = 48,
    ) -> None:
        """Initialize seasonal decomposition phase.
        
        Args:
            enable_seasonality: Master switch for seasonality.
            seasonal_period_default: Default seasonal period.
            seasonal_use_stl: Use STL instead of FFT.
            seasonal_min_points: Minimum points required.
        """
        self._enabled = enable_seasonality
        self._period_default = seasonal_period_default
        self._use_stl = seasonal_use_stl
        self._min_points = seasonal_min_points
        
        # Initialize decomposer
        if seasonal_use_stl:
            self._decomposer = STLDecomposer(period=seasonal_period_default)
            self._decomposer_name = "stl"
        else:
            self._decomposer = FFTSeasonalityDetector(
                min_period=4,
                max_period=min(100, seasonal_period_default * 4),
            )
            self._decomposer_name = "fft"
    
    @property
    def name(self) -> str:
        return "seasonal_decomposition"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute seasonal decomposition phase.
        
        Args:
            ctx: Pipeline context with raw values.
        
        Returns:
            Updated context with seasonal component removed from values.
        """
        start_time = time.perf_counter()
        
        # Log phase start
        logger.info(
            "seasonal_decomposition_phase_start",
            extra={
                "phase": self.name,
                "series_id": ctx.series_id,
                "event": "phase_start",
            },
        )
        
        # Early exit if disabled
        if not self._enabled:
            return ctx.with_field(
                seasonal_period_detected=None,
                seasonal_component_removed=False,
            )
        
        # Early exit if insufficient data
        # Values pre-sanitized by SanitizePhase[0] — see pipeline_executor.py
        if len(ctx.values) < self._min_points:
            logger.warning(
                "seasonal_insufficient_data",
                extra={
                    "phase": self.name,
                    "series_id": ctx.series_id,
                    "event": "WARNING",
                    "reason": "insufficient_data_for_seasonal_decomposition",
                    "required": self._min_points,
                    "received": len(ctx.values),
                    "action_taken": "skip_seasonal_decomposition",
                },
            )
            return ctx.with_field(
                seasonal_period_detected=None,
                seasonal_component_removed=False,
            )
        
        # Get period from SeriesValuesStore or use default
        period = self._get_period(ctx)
        
        # Decompose
        result = self._decompose(ctx.values, period)
        
        if result is None:
            # Decomposition failed or no cycle detected
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "seasonal_decomposition_phase_complete",
                extra={
                    "phase": self.name,
                    "series_id": ctx.series_id,
                    "event": "phase_complete",
                    "result": {
                        "seasonal_period_detected": None,
                        "seasonal_component_removed": False,
                        "decomposer_used": self._decomposer_name,
                    },
                    "duration_ms": round(duration_ms, 2),
                },
            )
            return ctx.with_field(
                seasonal_period_detected=None,
                seasonal_component_removed=False,
            )
        
        trend, seasonal, residual = result
        
        # Replace values with residual
        new_values = residual
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Log phase completion
        logger.info(
            "seasonal_decomposition_phase_complete",
            extra={
                "phase": self.name,
                "series_id": ctx.series_id,
                "event": "phase_complete",
                "result": {
                    "seasonal_period_detected": period,
                    "seasonal_component_removed": True,
                    "decomposer_used": self._decomposer_name,
                    "original_values_count": len(ctx.values),
                    "residual_values_count": len(new_values),
                },
                "duration_ms": round(duration_ms, 2),
            },
        )
        
        return ctx.with_field(
            values=new_values,
            seasonal_period_detected=period,
            seasonal_component_removed=True,
            seasonal_trend=trend,
            seasonal_component=seasonal,
        )
    
    def _get_period(self, ctx: PipelineContext) -> int:
        """Get seasonal period from SeriesValuesStore or default."""
        # Try to get from SeriesValuesStore
        orchestrator = ctx.orchestrator
        if hasattr(orchestrator, '_series_values_store'):
            try:
                store = orchestrator._series_values_store
                if hasattr(store, 'get_metadata'):
                    metadata = store.get_metadata(ctx.series_id)
                    if metadata and 'seasonal_period' in metadata:
                        return int(metadata['seasonal_period'])
            except Exception as e:
                logger.debug(f"failed_to_get_period_from_store: {e}")
        
        # Use default
        return self._period_default
    
    def _decompose(
        self,
        values: List[float],
        period: int,
    ) -> Optional[Tuple[List[float], List[float], List[float]]]:
        """Decompose time series using configured decomposer."""
        try:
            if self._use_stl and isinstance(self._decomposer, STLDecomposer):
                if not self._decomposer.available:
                    # Fallback to FFT
                    logger.warning(
                        "seasonal_stl_unavailable_fallback_fft",
                        extra={
                            "event": "WARNING",
                            "reason": "stl_decomposer_unavailable",
                            "action_taken": "fallback_to_fft",
                        },
                    )
                    fallback = FFTSeasonalityDetector(
                        min_period=4,
                        max_period=min(100, period * 4),
                    )
                    return fallback.decompose(values)
                
                return self._decomposer.decompose(values)
            else:
                return self._decomposer.decompose(values)
        
        except Exception as e:
            logger.error(
                "seasonal_decomposition_error",
                extra={
                    "phase": self.name,
                    "event": "PHASE_ERROR",
                    "error": str(e),
                    "action_taken": "return_none",
                },
            )
            return None
