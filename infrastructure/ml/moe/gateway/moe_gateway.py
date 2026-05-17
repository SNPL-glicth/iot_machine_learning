"""MoEGateway — Gateway principal de arquitectura Mixture of Experts (legacy).

Implementa PredictionPort del dominio usando MoE internamente.
Coordena: gating → dispatch → fusion → metadata.

Refactorizado: SRP - solo orquestación, lógica delegada a servicios.
DEPRECATED: Usar MoEPredictionEngine como engine dentro del pipeline.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, List, Any

from iot_machine_learning.domain.ports.prediction_port import PredictionPort
from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow

from ..registry.expert_registry import ExpertRegistry
from ..gating.base import GatingNetwork
from ..fusion.sparse_fusion import SparseFusionLayer

from .expert_dispatcher import ExpertDispatcher
from .prediction_enricher import PredictionEnricher, MoEMetadata


class MoEGateway(PredictionPort):
    """Gateway de Mixture of Experts implementando PredictionPort.
    
    Flujo de ejecución:
    1. ContextEncoderService.encode() → ContextVector
    2. gating.route() → GatingProbs
    3. GatingProbs.get_top_k() → k expertos
    4. ExpertDispatcher.dispatch() → ejecutar expertos
    5. SparseFusionLayer.fuse() → Prediction
    6. PredictionEnricher.enrich() → Prediction con metadata MoE
    Args:
        registry: Catálogo de expertos.
        gating: Estrategia de routing.
        fusion: Capa de fusión.
        fallback_engine: Engine original para fallback.
        sparsity_k: Número de expertos a ejecutar (default 2).
    """
    
    def __init__(
        self,
        registry: ExpertRegistry,
        gating: GatingNetwork,
        fusion: SparseFusionLayer,
        fallback_engine: PredictionPort,
        sparsity_k: int = 2,
    ):
        self._registry = registry
        self._gating = gating
        self._fusion = fusion
        self._fallback = fallback_engine
        self._sparsity_k = sparsity_k

        # Servicios extraídos (SRP)
        self._dispatcher = ExpertDispatcher(registry)
        self._enricher = PredictionEnricher()
    
    @property
    def name(self) -> str:
        """Nombre del gateway (para PredictionPort)."""
        return "moe_gateway"
    
    def predict(self, window: SensorWindow) -> Prediction:
        """Genera predicción usando MoE.

        Args:
            window: Ventana temporal de lecturas.

        Returns:
            Prediction del dominio.
        """
        return self._predict_moe(window)

    def _predict_moe(self, window: SensorWindow) -> Prediction:
        """Flujo completo de predicción MoE.
        
        Args:
            window: Ventana de lecturas.
            
        Returns:
            Prediction con metadata MoE.
        """
        start_time = time.perf_counter()

        # 1. Gating: distribución sobre expertos
        from iot_machine_learning.domain.model.context_vector import ContextVector
        values = [r.value for r in window.readings]
        context = ContextVector(
            regime="stable",
            domain="iot",
            n_points=len(values),
            signal_features={"mean": sum(values) / max(len(values), 1)},
        )
        gating_result = self._gating.route(context)
        
        # 3. Select top-k expertos
        selected_experts = gating_result.get_top_k(self._sparsity_k)
        
        # 4. Dispatch: ejecutar expertos seleccionados
        expert_outputs = self._dispatcher.dispatch(selected_experts, window)
        
        # 5. Fusion: combinar resultados
        # Extraer pesos de los expertos ejecutados
        fusion_weights = {
            eid: gating_result.probabilities[eid]
            for eid in expert_outputs.keys()
        }
        
        prediction = self._fusion.fuse(expert_outputs, fusion_weights)
        
        # 6. Enrich metadata
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        
        moe_metadata = MoEMetadata(
            selected_experts=list(expert_outputs.keys()),
            sparsity_k=len(expert_outputs),
            gating_probs=dict(gating_result.probabilities),
            fusion_weights=self._fusion.get_fusion_weights(fusion_weights).normalized,
            dominant_expert=max(fusion_weights.items(), key=lambda x: x[1])[0],
            total_latency_ms=total_latency_ms,
            moe_enabled=True,
        )
        
        return self._enricher.enrich(prediction, moe_metadata, window)
    
    def can_handle(self, n_points: int) -> bool:
        """Verifica si puede operar con n_points datos.
        
        Delega al fallback si MoE deshabilitado.
        
        Args:
            n_points: Número de puntos disponibles.
            
        Returns:
            True si puede generar predicción.
        """
        # Verificar capacidad directamente via registry (dispatcher no expone esto)
        from iot_machine_learning.domain.entities.sensor_reading import SensorWindow, SensorReading
        dummy_window = SensorWindow(
            series_id="_can_handle_check",
            readings=[SensorReading(value=0.0, timestamp=float(i)) for i in range(n_points)],
        )
        for expert_id in self._registry.list_all():
            expert = self._registry.get(expert_id)
            if expert and expert.can_handle(dummy_window):
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del gateway.
        
        Returns:
            Dict con métricas de operación.
        """
        return self._enricher.get_stats(
            sparsity_k=self._sparsity_k,
            registry_size=len(self._registry),
            gating_type=self._gating.__class__.__name__,
        )
