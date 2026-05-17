"""MoEPredictionEngine — motor de predicción MoE como engine del pipeline.

Implementa PredictionEngine (como Kalman o Taylor) para integración nativa
en el pipeline cognitivo. Recibe FeatureContext cuando está disponible;
si no, construye uno básico desde los valores (fallback standalone).

Diseño SOLID:
- SRP: solo orquesta gating + dispatch + fusion.
- DIP: depende de abstracciones (GatingStrategy, ExpertPort, PredictionPort).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Any

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
from ..gating.contextual_regime import ContextualRegimeGating
from ..gating.tree_gating import TreeGatingNetwork
from ..fusion.discrepancy_aware import DiscrepancyAwareFusion
from ..gateway.expert_dispatcher import ExpertDispatcher
from ..gateway.prediction_enricher import PredictionEnricher, MoEMetadata
from ..ab.moe_ab_logger import MoEABLogger, ABLogEntry


class MoEPredictionEngine(PredictionEngine):
    """Motor MoE integrado como engine de predicción del pipeline.

    Args:
        registry: Catálogo de expertos.
        gating: Estrategia de routing (GatingStrategy).
        fusion: Capa de fusión (DiscrepancyAwareFusion).
        fallback_engine: Engine para fallback si no hay expertos.
        sparsity_k: Número de expertos top-k a ejecutar.
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        gating: Optional[GatingStrategy] = None,
        fusion: Optional[DiscrepancyAwareFusion] = None,
        fallback_engine: Optional[PredictionPort] = None,
        sparsity_k: int = 2,
        shadow_gating: Optional[TreeGatingNetwork] = None,
        ab_logger: Optional[MoEABLogger] = None,
        ab_cell: str = "B",
        metrics_exporter=None,
        alert_service=None,
    ) -> None:
        self._registry = registry
        self._gating = gating or ContextualRegimeGating(
            expert_ids=registry.list_all(),
        )
        self._fusion = fusion or DiscrepancyAwareFusion()
        self._fallback = fallback_engine
        self._sparsity_k = sparsity_k
        self._shadow_gating = shadow_gating
        self._ab_logger = ab_logger
        self._ab_cell = ab_cell
        self._metrics_exporter = metrics_exporter
        self._alert_service = alert_service

        # Leer timeout desde config externa
        timeout_ms = 200
        try:
            from ..config.moe_config import DISPATCH_TIMEOUT_MS
            timeout_ms = DISPATCH_TIMEOUT_MS
        except Exception:
            pass

        self._dispatcher = ExpertDispatcher(registry, timeout_ms=timeout_ms)
        self._enricher = PredictionEnricher()

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
        """Genera predicción usando MoE.

        Si ``feature_context`` es None, construye uno básico desde los valores
        para operación standalone. Cuando el pipeline lo provee, no recalcula.

        Args:
            values: Serie temporal.
            timestamps: Timestamps opcionales.
            feature_context: Contexto enriquecido del pipeline (opcional).
            series_id: Identificador de serie para métricas (opcional).

        Returns:
            PredictionResult con valor, confianza, trend y metadata.
        """
        if feature_context is None:
            feature_context = self._build_context_from_values(values)

        start_time = time.perf_counter()
        _sid = series_id or "unknown"

        # Si no hay expertos registrados, fallback inmediato
        if len(self._registry) == 0:
            self._record_fallback(_sid)
            return self._fallback_predict(values, timestamps, "empty_registry")

        # 1. Gating: distribución sobre expertos
        gating_result = self._gating.route(feature_context)
        selected_experts = gating_result.get_top_k(self._sparsity_k)
        self._record_regime(feature_context.regime)

        # 1b. Shadow gating: TreeGatingNetwork en modo observador
        shadow_metadata: Dict[str, Any] = {}
        if self._shadow_gating is not None:
            try:
                from iot_machine_learning.domain.model.context_vector import ContextVector
                shadow_ctx = ContextVector(
                    regime=feature_context.regime,
                    domain="iot",
                    n_points=len(values),
                    signal_features={
                        "mean": feature_context.mean,
                        "std": feature_context.std,
                        "slope": feature_context.slope,
                    },
                )
                shadow_probs = self._shadow_gating.route(shadow_ctx)
                max_diff = 0.0
                for eid in set(gating_result.probabilities.keys()) | set(shadow_probs.probabilities.keys()):
                    p1 = gating_result.probabilities.get(eid, 0.0)
                    p2 = shadow_probs.probabilities.get(eid, 0.0)
                    max_diff = max(max_diff, abs(p1 - p2))
                shadow_metadata = {
                    "shadow_gating": {
                        "top_expert": shadow_probs.top_expert,
                        "max_prob_diff": round(max_diff, 4),
                        "shadow_entropy": round(shadow_probs.entropy, 4),
                    }
                }
            except Exception as exc:
                import logging
                logging.getLogger("moe.shadow").warning(
                    "shadow_gating_failed", extra={"error": str(exc)}
                )
                shadow_metadata = {"shadow_gating": {"error": str(exc)}}

        # 2. Dispatch: ejecutar expertos seleccionados
        window = self._make_window(values, timestamps)
        expert_outputs = self._dispatcher.dispatch(selected_experts, window)

        # Si no hay outputs válidos, fallback
        if not expert_outputs:
            self._record_fallback(_sid)
            return self._fallback_predict(values, timestamps, "no_experts_available")

        # 3. Fusion: combinar resultados
        fusion_weights = {
            eid: gating_result.probabilities[eid]
            for eid in expert_outputs.keys()
        }
        prediction = self._fusion.fuse(expert_outputs, fusion_weights)

        # 4. Enrich metadata
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        dominant = max(fusion_weights.items(), key=lambda x: x[1])[0]

        # Métricas: latencia por experto
        for eid, out in expert_outputs.items():
            self._record_expert_latency(eid, out.latency_ms, _sid)

        # Métrica: discrepancia
        std_pred = self._fusion._std_of_predictions(expert_outputs)
        self._record_discrepancy(_sid, std_pred)

        moe_metadata = MoEMetadata(
            selected_experts=list(expert_outputs.keys()),
            sparsity_k=len(expert_outputs),
            gating_probs=dict(gating_result.probabilities),
            fusion_weights=self._fusion.get_fusion_weights(fusion_weights).normalized,
            dominant_expert=dominant,
            total_latency_ms=total_latency_ms,
            moe_enabled=True,
        )

        enriched = self._enricher.enrich(prediction, moe_metadata, window)

        # Merge shadow metadata (inmutable: crear nuevo Prediction)
        if shadow_metadata:
            from dataclasses import replace
            merged_meta = {**(enriched.metadata or {}), **shadow_metadata}
            enriched = replace(enriched, metadata=merged_meta)

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
                expert_weights=dict(gating_result.probabilities),
                selected_experts=list(expert_outputs.keys()),
                dominant_expert=dominant,
                metadata=shadow_metadata,
            )
            self._ab_logger.log_prediction(entry)

        return PredictionResult(
            predicted_value=enriched.predicted_value,
            confidence=enriched.confidence_score,
            trend=enriched.trend,
            metadata=enriched.metadata,
        )

    def can_handle(self, n_points: int) -> bool:
        """Verifica si puede operar con n_points datos.

        Delega a fallback si no hay expertos capaces.
        """
        # Verificar expertos
        for expert_id in self._registry.list_all():
            expert = self._registry.get(expert_id)
            if expert and expert.can_handle(self._make_dummy_window(n_points)):
                return True
        # Fallback
        if self._fallback is not None:
            return self._fallback.can_handle(n_points)
        return False

    def as_port(self) -> "PredictionEnginePortBridge":
        """Devuelve bridge estándar, compatible con container."""
        from iot_machine_learning.infrastructure.ml.interfaces import (
            PredictionEnginePortBridge,
        )
        return PredictionEnginePortBridge(self)

    # ------------------------------------------------------------------
    # Pipeline-aware API (extensión de PredictionEngine)
    # ------------------------------------------------------------------

    def predict_with_context(
        self,
        values: List[float],
        timestamps: Optional[List[float]],
        feature_context: FeatureContext,
    ) -> PredictionResult:
        """Pipeline-aware: usa FeatureContext sin recalcular."""
        return self.predict(values, timestamps, feature_context=feature_context)

    # ------------------------------------------------------------------
    # Métricas y alertas
    # ------------------------------------------------------------------

    def _record_fallback(self, series_id: str) -> None:
        if self._metrics_exporter is not None:
            self._metrics_exporter.increment_moe_fallback()
        if self._alert_service is not None:
            self._alert_service.record_fallback()

    def _record_regime(self, regime: str) -> None:
        if self._metrics_exporter is not None:
            self._metrics_exporter.record_moe_regime(regime)

    def _record_expert_latency(self, expert_id: str, latency_ms: float, series_id: str) -> None:
        if self._metrics_exporter is not None:
            self._metrics_exporter.record_moe_expert_latency(expert_id, latency_ms, series_id)

    def _record_discrepancy(self, series_id: str, score: float) -> None:
        if self._metrics_exporter is not None:
            self._metrics_exporter.record_moe_discrepancy(series_id, score)
        if self._alert_service is not None:
            self._alert_service.record_discrepancy(score)

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _build_context_from_values(self, values: List[float]) -> FeatureContext:
        """Construye FeatureContext básico desde valores (standalone fallback)."""
        import statistics

        n = len(values)
        mean = sum(values) / n if n > 0 else 0.0
        std = statistics.stdev(values) if n >= 2 else 0.0

        # Estimación simple de slope (diferencia último - primero / intervalo)
        slope = 0.0
        if n >= 2:
            slope = (values[-1] - values[0]) / max(n - 1, 1)

        # Estimación simple de noise_ratio
        noise_ratio = std / abs(mean) if abs(mean) > 1e-9 else 0.0

        # Clasificación de régimen simple
        if noise_ratio > 0.5:
            regime = "noisy"
        elif abs(slope) > 0.01 * abs(mean) if abs(mean) > 1e-9 else abs(slope) > 0.01:
            regime = "trending"
        elif std > 0.5:
            regime = "volatile"
        else:
            regime = "stable"

        return FeatureContext(
            regime=regime,
            mean=mean,
            std=std,
            slope=slope,
            curvature=0.0,
            noise_ratio=noise_ratio,
            stability=0.0,
            hampel_outlier_mask=[],
            spatial_correlation_score=0.0,
        )

    def _fallback_predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]],
        reason: str,
    ) -> PredictionResult:
        """Fallback cuando MoE no puede operar."""
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
    ) -> SensorWindow:
        from iot_machine_learning.domain.entities.iot.sensor_reading import (
            SensorWindow, Reading,
        )
        ts = timestamps if timestamps is not None else list(range(len(values)))
        readings = [
            Reading(series_id="moe", value=v, timestamp=t)
            for v, t in zip(values, ts)
        ]
        return SensorWindow(series_id="moe", readings=readings)

    @staticmethod
    def _make_dummy_window(n_points: int) -> SensorWindow:
        from iot_machine_learning.domain.entities.iot.sensor_reading import (
            SensorWindow, Reading,
        )
        readings = [
            Reading(series_id="_check", value=0.0, timestamp=float(i))
            for i in range(n_points)
        ]
        return SensorWindow(series_id="_check", readings=readings)
