"""Memory Phase — Cognitive memory integration.

Integrates semantic event building, anomaly memory, and operational patterns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import PipelineContext

try:
    from ...memory import SemanticEventBuilder, AnomalyMemoryStore
except (ImportError, ModuleNotFoundError):
    SemanticEventBuilder = None  # type: ignore[assignment,misc]
    AnomalyMemoryStore = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class MemoryPhase:
    """Phase: Cognitive memory integration.
    
    Builds semantic events, stores anomalies in memory, and provides
    operational pattern context for downstream phases.
    """
    
    def __init__(
        self,
        semantic_event_builder: Optional[Any] = None,
        anomaly_memory_store: Optional[Any] = None,
    ) -> None:
        """Initialize memory phase.
        
        Args:
            semantic_event_builder: Optional SemanticEventBuilder instance.
            anomaly_memory_store: Optional AnomalyMemoryStore instance.
        """
        self._semantic_event_builder = semantic_event_builder
        if SemanticEventBuilder is not None and self._semantic_event_builder is None:
            self._semantic_event_builder = SemanticEventBuilder()
        
        self._anomaly_memory_store = anomaly_memory_store
        if AnomalyMemoryStore is not None and self._anomaly_memory_store is None:
            self._anomaly_memory_store = AnomalyMemoryStore()
    
    @property
    def name(self) -> str:
        return "memory"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute memory phase.
        
        Args:
            ctx: Pipeline context with cognitive metrics.
        
        Returns:
            Updated context with memory context.
        """
        # Skip if no memory components available
        if ctx.memory_registry is None:
            return ctx
        
        try:
            # Build semantic event from context
            semantic_event = None
            if self._semantic_event_builder is not None:
                try:
                    semantic_event = self._semantic_event_builder.build_event(
                        sensor_id=ctx.series_id,
                        regime=ctx.regime,
                        anomaly_score=getattr(ctx.profile, "z_score", 0.0) if ctx.profile else 0.0,
                        confidence=ctx.fused_confidence or 0.0,
                        timestamp=ctx.timestamps[-1] if ctx.timestamps else None,
                    )
                    
                    # Store in memory registry
                    if semantic_event and ctx.memory_registry:
                        ctx.memory_registry.register_event(semantic_event)
                except Exception as e:
                    logger.debug(f"semantic_event_building_failed: {e}")
            
            # Store anomaly in memory if detected
            if self._anomaly_memory_store is not None and ctx.profile:
                try:
                    z_score = getattr(ctx.profile, "z_score", 0.0)
                    if abs(z_score) > 2.5:  # Anomaly threshold
                        self._anomaly_memory_store.store_anomaly(
                            sensor_id=ctx.series_id,
                            anomaly_score=z_score,
                            regime=ctx.regime,
                            timestamp=ctx.timestamps[-1] if ctx.timestamps else None,
                        )
                except Exception as e:
                    logger.debug(f"anomaly_memory_storage_failed: {e}")
            
            # Log memory summary
            logger.debug(
                "memory_phase_completed",
                extra={
                    "series_id": ctx.series_id,
                    "semantic_event_created": semantic_event is not None,
                    "anomaly_stored": abs(getattr(ctx.profile, "z_score", 0.0)) > 2.5 if ctx.profile else False,
                },
            )
            
            return ctx.with_field(
                semantic_event=semantic_event,
                memory_context={
                    "has_memory": True,
                    "event_created": semantic_event is not None,
                },
            )
        
        except Exception as e:
            logger.debug(f"memory_phase_failed: {e}")
            return ctx
