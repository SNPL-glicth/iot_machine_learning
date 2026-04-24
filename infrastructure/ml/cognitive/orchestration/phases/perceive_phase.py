"""Perceive Phase — MED-1 Refactoring.

Analyzes signal and extracts neighbor information.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
        
        return ctx.with_field(
            profile=profile,
            regime=regime_str,
            neighbor_trends=neighbor_trends,
            neighbors=neighbors,
            neighbor_values=neighbor_values_dict,
            plasticity_context=plasticity_context,
        )
