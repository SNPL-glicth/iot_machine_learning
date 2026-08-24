"""MoEPredictionEngine — Main MoE engine orchestrating gating, dispatch, and fusion."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.ports.expert_port import ExpertOutput
from iot_machine_learning.domain.ports.prediction_port import PredictionPort

from ..feature_context import FeatureContext
from ..registry import ExpertRegistry
from ..gating.strategy import GatingStrategy
from ..gating.contextual_regime_gating import ContextualRegimeGating
from ..fusion.discrepancy_aware import DiscrepancyAwareFusion

from .moe_context_builder import MoEContextBuilder
from .moe_gating_executor import MoEGatingExecutor
from .moe_fusion_pipeline import MoEFusionPipeline


class MoEPredictionEngine(PredictionEngine):
    """MoE prediction engine composed of modular components."""
    
    def __init__(
        self,
        registry: ExpertRegistry,
        gating: Optional[GatingStrategy] = None,
        fusion: Optional[DiscrepancyAwareFusion] = None,
        fallback_engine: Optional[PredictionPort] = None,
        sparsity_k: int = 2,
        shadow_gating=None,
        ab_logger=None,
        ab_cell: str = "B",
        metrics_exporter=None,
        alert_service=None,
    ) -> None:
        self._registry = registry
        self._fallback = fallback_engine
        self._sparsity_k = sparsity_k
        
        # Composed components
        self._context_builder = MoEContextBuilder()
        self._gating_executor = MoEGatingExecutor(
            registry=registry,
            gating=gating,
            shadow_gating=shadow_gating,
        )
        self._fusion_pipeline = MoEFusionPipeline(
            fusion=fusion,
            metrics_exporter=metrics_exporter,
            alert_service=alert_service,
            ab_logger=ab_logger,
            ab_cell=ab_cell,
        )
    
    @property
    def name(self) -> str:
        return "moe_engine"
    
    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
        feature_context: Optional[FeatureContext] = None,
        series_id: Optional[str] = None,
    ) -> PredictionResult:
        """Generate prediction using MoE."""
        if feature_context is None:
            feature_context = self._context_builder.build(values)
        
        start_time = time.perf_counter()
        _sid = series_id or "unknown"
        
        # Fallback if no experts
        if len(self._registry) == 0:
            return self._fallback_predict(values, timestamps, "empty_registry")
        
        # 1. Gating: select experts
        selected_experts, shadow_metadata = self._gating_executor.execute_gating(
            feature_context, self._sparsity_k
        )
        regime = self._gating_executor.get_regime(feature_context)
        
        # 2. Dispatch experts
        window = self._make_window(values, timestamps, series_id)
        expert_outputs = self._gating_executor.dispatch_experts(selected_experts, window)
        
        # Fallback if no valid outputs
        if not expert_outputs:
            return self._fallback_predict(values, timestamps, "no_experts_available")
        
        # 3. Fusion + enrichment
        gating_probs = self._gating_executor._gating.route(feature_context).probabilities
        enriched = self._fusion_pipeline.fuse_and_enrich(
            expert_outputs=expert_outputs,
            gating_probs=gating_probs,
            window=window,
            start_time=start_time,
            feature_context=feature_context,
        )
        
        # Merge shadow metadata
        enriched = self._fusion_pipeline.merge_shadow_metadata(enriched, shadow_metadata)
        
        return PredictionResult(
            predicted_value=enriched.predicted_value,
            confidence=enriched.confidence_score,
            trend=enriched.trend,
            metadata=enriched.metadata,
        )
    
    def can_handle(self, n_points: int) -> bool:
        for expert_id in self._registry.list_all():
            expert = self._registry.get(expert_id)
            if expert and expert.can_handle(self._make_dummy_window(n_points)):
                return True
        if self._fallback is not None:
            return self._fallback.can_handle(n_points)
        return False
    
    def as_port(self) -> "PredictionEnginePortBridge":
        from iot_machine_learning.infrastructure.ml.interfaces import PredictionEnginePortBridge
        return PredictionEnginePortBridge(self)
    
    def predict_with_context(
        self,
        values: List[float],
        timestamps: Optional[List[float]],
        feature_context: FeatureContext,
    ) -> PredictionResult:
        return self.predict(values, timestamps, feature_context=feature_context)
    
    def _fallback_predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]],
        reason: str,
    ) -> PredictionResult:
        if self._fallback is not None:
            window = self._make_window(values, timestamps)
            pred = self._fallback.predict(window)
            return PredictionResult(
                predicted_value=pred.predicted_value,
                confidence=pred.confidence_score * 0.8,
                trend=pred.trend,
                metadata={
                    "moe_fallback": True,
                    "fallback_reason": reason,
                    "fallback_engine": self._fallback.name,
                },
            )
        return PredictionResult(
            predicted_value=values[-1] if values else None,
            confidence=0.0,
            trend="unknown",
            metadata={"moe_fallback": True, "fallback_reason": reason},
        )
    
    @staticmethod
    def _make_window(
        values: List[float],
        timestamps: Optional[List[float]],
        series_id: Optional[str] = None,
    ) -> SensorWindow:
        from iot_machine_learning.domain.entities.iot.sensor_reading import SensorWindow, Reading
        sid = series_id or "moe"
        ts = timestamps if timestamps is not None else list(range(len(values)))
        readings = [Reading(series_id=sid, value=v, timestamp=t) for v, t in zip(values, ts)]
        return SensorWindow(series_id=sid, readings=readings)
    
    @staticmethod
    def _make_dummy_window(n_points: int) -> SensorWindow:
        from iot_machine_learning.domain.entities.iot.sensor_reading import SensorWindow, Reading
        readings = [Reading(series_id="_check", value=0.0, timestamp=float(i)) for i in range(n_points)]
        return SensorWindow(series_id="_check", readings=readings)