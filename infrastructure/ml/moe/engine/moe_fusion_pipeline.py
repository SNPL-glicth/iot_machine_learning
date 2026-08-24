"""MoE Fusion Pipeline: Handles fusion, enrichment, metrics, and A/B logging."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.ports.expert_port import ExpertOutput

from ..fusion.discrepancy_aware import DiscrepancyAwareFusion
from ..gateway.prediction_enricher import PredictionEnricher, MoEMetadata
from ..ab.moe_ab_logger import MoEABLogger, ABLogEntry
from ..feature_context import FeatureContext


class MoEFusionPipeline:
    """Fuses expert outputs, enriches metadata, records metrics, and logs A/B."""
    
    def __init__(
        self,
        fusion: Optional[DiscrepancyAwareFusion] = None,
        enricher: Optional[PredictionEnricher] = None,
        metrics_exporter=None,
        alert_service=None,
        ab_logger: Optional[MoEABLogger] = None,
        ab_cell: str = "B",
    ):
        self._fusion = fusion or DiscrepancyAwareFusion()
        self._enricher = enricher or PredictionEnricher()
        self._metrics_exporter = metrics_exporter
        self._alert_service = alert_service
        self._ab_logger = ab_logger
        self._ab_cell = ab_cell
    
    def fuse_and_enrich(
        self,
        expert_outputs: Dict[str, ExpertOutput],
        gating_probs: Dict[str, float],
        window,
        start_time: float,
        feature_context: FeatureContext,
    ) -> Prediction:
        """Fuse expert outputs, enrich with metadata, and return final Prediction."""
        # Fusion weights from gating probabilities
        fusion_weights = {
            eid: gating_probs[eid]
            for eid in expert_outputs.keys()
        }
        
        prediction = self._fusion.fuse(expert_outputs, fusion_weights)
        
        # Enrich metadata
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        dominant = max(fusion_weights.items(), key=lambda x: x[1])[0]
        
        # Metrics
        if self._metrics_exporter is not None:
            for eid, out in expert_outputs.items():
                self._metrics_exporter.record_moe_expert_latency(eid, out.latency_ms, window.series_id)
        
        std_pred = self._fusion._std_of_predictions(expert_outputs)
        if self._metrics_exporter is not None:
            self._metrics_exporter.record_moe_discrepancy(window.series_id, std_pred)
        if self._alert_service is not None:
            self._alert_service.record_discrepancy(std_pred)
        
        moe_metadata = MoEMetadata(
            selected_experts=list(expert_outputs.keys()),
            sparsity_k=len(expert_outputs),
            gating_probs=dict(gating_probs),
            fusion_weights=self._fusion.get_fusion_weights(fusion_weights).normalized,
            dominant_expert=dominant,
            total_latency_ms=total_latency_ms,
            moe_enabled=True,
        )
        
        enriched = self._enricher.enrich(prediction, moe_metadata, window)
        
        # A/B logging
        if self._ab_logger is not None:
            from datetime import datetime, timezone
            entry = ABLogEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                cell=self._ab_cell,
                engine_used="moe_engine",
                prediction_value=enriched.predicted_value,
                confidence=enriched.confidence_score,
                latency_ms=total_latency_ms,
                regime=feature_context.regime,
                expert_weights=dict(gating_probs),
                selected_experts=list(expert_outputs.keys()),
                dominant_expert=dominant,
            )
            self._ab_logger.log_prediction(entry)
        
        return enriched
    
    def merge_shadow_metadata(self, prediction: Prediction, shadow_metadata: Dict[str, Any]) -> Prediction:
        """Merge shadow gating metadata into prediction."""
        if shadow_metadata:
            from dataclasses import replace
            merged_meta = {**(prediction.metadata or {}), **shadow_metadata}
            return replace(prediction, metadata=merged_meta)
        return prediction