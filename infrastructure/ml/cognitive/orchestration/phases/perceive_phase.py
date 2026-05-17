"""Perceive Phase — MED-1 Refactoring.

Analyzes signal and extracts neighbor information.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from iot_machine_learning.domain.value_objects.industrial_event import EventContext

if TYPE_CHECKING:
    from . import PipelineContext

logger = logging.getLogger(__name__)


class PerceivePhase:
    """Phase 1: Signal analysis and correlation enrichment."""
    
    @property
    def name(self) -> str:
        return "perceive"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute signal perception."""
        orchestrator = ctx.orchestrator
        
        # Signal analysis
        profile = orchestrator._analyzer.analyze(ctx.values, ctx.timestamps)
        regime_str = profile.regime.value if hasattr(profile.regime, 'value') else str(profile.regime)

        # Extraer hour_of_day del último timestamp para contexto temporal
        hour_of_day = None
        if ctx.timestamps:
            try:
                import datetime as _dt
                hour_of_day = _dt.datetime.fromtimestamp(ctx.timestamps[-1]).hour
            except Exception:
                pass

        # Correlation enrichment
        neighbor_trends = {}
        neighbors = []
        neighbor_values_dict = {}
        
        if orchestrator._correlation_port and ctx.series_id != "unknown":
            try:
                neighbors = orchestrator._correlation_port.get_correlated_series(
                    ctx.series_id, max_neighbors=3
                )
                if neighbors:
                    neighbor_ids = [n[0] for n in neighbors]
                    neighbor_values_dict = orchestrator._correlation_port.get_recent_values_multi(
                        neighbor_ids, window=5
                    )
                    for nid, nvals in neighbor_values_dict.items():
                        if len(nvals) >= 2:
                            slope = (nvals[-1] - nvals[0]) / max(len(nvals) - 1, 1)
                            neighbor_trends[nid] = (
                                "up" if slope > 0.1 else 
                                "down" if slope < -0.1 else 
                                "stable"
                            )
            except Exception as e:
                logger.debug(f"correlation_enrichment_failed: {e}")
        
        # Create plasticity context if enabled
        plasticity_context = None
        if orchestrator._enable_advanced_plasticity and orchestrator._plasticity_coordinator:
            plasticity_context = orchestrator._plasticity_coordinator.create_signal_context(
                profile, ctx.series_id
            )

        # Load SensorProfile if the orchestrator has the repository
        sensor_profile = None
        profile_repo = getattr(orchestrator, '_sensor_profile_repository', None)
        if profile_repo is not None and ctx.series_id != "unknown":
            try:
                sensor_profile = profile_repo.get_by_series_id(ctx.series_id)
            except Exception as e:
                logger.debug(f"sensor_profile_load_failed series={ctx.series_id}: {e}")

        # Fase 3: detectar evento industrial
        from iot_machine_learning.infrastructure.ml.moe.events.industrial_event_detector import detect_industrial_event
        event_ctx = EventContext.none()
        try:
            event_ctx = detect_industrial_event(ctx.values or [], list(getattr(ctx, "sanitization_flags", [])), sensor_profile)
            if event_ctx.is_active:
                logger.info("industrial_event_detected", extra={"series": ctx.series_id, "event": event_ctx.detected_event.value, "conf": round(event_ctx.event_confidence, 2)})
        except Exception as e:
            logger.debug(f"event_detection_failed: {e}")
        # Reclasificar régimen con contexto temporal si hay perfil
        if hour_of_day is not None and sensor_profile is not None:
            from iot_machine_learning.domain.entities.series.structural_analysis import _classify_regime as _cr
            enriched = _cr(getattr(profile, "noise_ratio", 0.0), getattr(profile, "slope", 0.0), getattr(profile, "std", 0.0), getattr(profile, "mean", 0.0), hour_of_day)
            regime_str = enriched.value if hasattr(enriched, 'value') else str(enriched)
            logger.debug(f"perceive_hour_of_day series={ctx.series_id} hour={hour_of_day} regime={regime_str}")

        # Build FeatureContext for downstream MoE consumption (pipeline-aware)
        from iot_machine_learning.infrastructure.ml.moe.feature_context import FeatureContext
        feature_ctx = FeatureContext.from_structural_analysis_with_profile(
            regime=regime_str,
            mean=getattr(profile, "mean", 0.0),
            std=getattr(profile, "std", 0.0),
            slope=getattr(profile, "slope", 0.0),
            curvature=getattr(profile, "curvature", 0.0),
            noise_ratio=getattr(profile, "noise_ratio", 0.0),
            stability=getattr(profile, "stability", 0.0),
            hampel_outlier_mask=[],
            spatial_correlation_score=(len(neighbors) / 3.0) if neighbors else 0.0,
            sensor_profile=sensor_profile,
            event_context=event_ctx,
        )

        return ctx.with_field(
            profile=profile,
            regime=regime_str,
            neighbor_trends=neighbor_trends,
            neighbors=neighbors,
            neighbor_values=neighbor_values_dict,
            plasticity_context=plasticity_context,
            feature_context=feature_ctx,
        )
