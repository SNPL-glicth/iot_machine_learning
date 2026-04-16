"""MoEGateway — Gateway principal de arquitectura Mixture of Experts.

Implementa PredictionPort del dominio usando MoE internamente.
Coordina: encoding → gating → dispatch → fusion → metadata.

Feature flag ML_MOE_ENABLED: si false, delega a engine original.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

from domain.ports.prediction_port import PredictionPort
from domain.ports.expert_port import ExpertPort, ExpertOutput
from domain.entities.prediction import Prediction
from domain.entities.sensor_reading import SensorWindow
from domain.model.context_vector import ContextVector

from ..registry.expert_registry import ExpertRegistry
from ..gating.base import GatingNetwork
from ..fusion.sparse_fusion import SparseFusionLayer, FusionWeights


@dataclass(frozen=True)
class MoEMetadata:
    """Metadata de ejecución MoE para trazabilidad.
    
    Attributes:
        selected_experts: IDs de expertos ejecutados.
        sparsity_k: Número de expertos activos.
        gating_probs: Probabilidades del gating.
        fusion_weights: Pesos usados en fusión.
        dominant_expert: Experto con mayor peso.
        total_latency_ms: Tiempo total de ejecución.
        moe_enabled: Si MoE estaba activo.
    """
    selected_experts: List[str]
    sparsity_k: int
    gating_probs: Dict[str, float]
    fusion_weights: Dict[str, float]
    dominant_expert: str
    total_latency_ms: float
    moe_enabled: bool


class MoEGateway(PredictionPort):
    """Gateway de Mixture of Experts implementando PredictionPort.
    
    Flujo de ejecución:
    1. encode_context() → ContextVector
    2. gating.route() → GatingProbs
    3. select_top_k() → k expertos
    4. dispatch() → ejecutar expertos
    5. fusion.fuse() → Prediction
    6. enrich_metadata() → Prediction con metadata MoE
    
    Feature flag ML_MOE_ENABLED:
    - true: Usa arquitectura MoE completa
    - false: Delega a fallback_engine (modo compatibilidad)
    
    Args:
        registry: Catálogo de expertos.
        gating: Estrategia de routing.
        fusion: Capa de fusión.
        fallback_engine: Engine original para modo compatibilidad.
        sparsity_k: Número de expertos a ejecutar (default 2).
        moe_enabled: Feature flag para activar/desactivar MoE.
    """
    
    def __init__(
        self,
        registry: ExpertRegistry,
        gating: GatingNetwork,
        fusion: SparseFusionLayer,
        fallback_engine: PredictionPort,
        sparsity_k: int = 2,
        moe_enabled: bool = True,
    ):
        self._registry = registry
        self._gating = gating
        self._fusion = fusion
        self._fallback = fallback_engine
        self._sparsity_k = sparsity_k
        self._moe_enabled = moe_enabled
    
    @property
    def name(self) -> str:
        """Nombre del gateway (para PredictionPort)."""
        return "moe_gateway"
    
    def predict(self, window: SensorWindow) -> Prediction:
        """Genera predicción usando MoE o fallback según feature flag.
        
        Args:
            window: Ventana temporal de lecturas.
            
        Returns:
            Prediction del dominio.
        """
        if not self._moe_enabled:
            # Feature flag desactivado: delegar a engine original
            return self._fallback.predict(window)
        
        return self._predict_moe(window)
    
    def _predict_moe(self, window: SensorWindow) -> Prediction:
        """Flujo completo de predicción MoE.
        
        Args:
            window: Ventana de lecturas.
            
        Returns:
            Prediction con metadata MoE.
        """
        start_time = time.perf_counter()
        
        # 1. Encode context
        context = self._encode_context(window)
        
        # 2. Gating: distribución sobre expertos
        gating_result = self._gating.route(context)
        
        # 3. Select top-k expertos
        selected_experts = gating_result.get_top_k(self._sparsity_k)
        
        # 4. Dispatch: ejecutar expertos seleccionados
        expert_outputs = self._dispatch(selected_experts, window)
        
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
        
        return self._enrich_prediction(prediction, moe_metadata, window)
    
    def _encode_context(self, window: SensorWindow) -> ContextVector:
        """Codifica ventana a vector de contexto.
        
        Args:
            window: Ventana de datos.
            
        Returns:
            ContextVector para gating.
        """
        # Extraer valores y features simples
        values = [r.value for r in window.readings]
        
        if not values:
            return ContextVector(
                regime="stable",
                domain="iot",
                n_points=0,
                signal_features={},
            )
        
        # Calcular features básicos
        mean_val = sum(values) / len(values)
        
        # Estimación simple de std
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_val = variance ** 0.5
        
        # Estimación simple de slope (diferencia últimos puntos)
        if len(values) >= 2:
            slope = values[-1] - values[-2]
        else:
            slope = 0.0
        
        # Clasificación simple de régimen
        if std_val / abs(mean_val) > 0.2 if mean_val != 0 else False:
            regime = "volatile"
        elif abs(slope) > std_val * 0.5:
            regime = "trending"
        else:
            regime = "stable"
        
        return ContextVector(
            regime=regime,
            domain="iot",  # Default
            n_points=len(values),
            signal_features={
                "mean": mean_val,
                "std": std_val,
                "slope": slope,
            },
        )
    
    def _dispatch(
        self,
        expert_ids: List[str],
        window: SensorWindow
    ) -> Dict[str, ExpertOutput]:
        """Ejecuta expertos seleccionados.
        
        Args:
            expert_ids: IDs de expertos a ejecutar.
            window: Ventana de datos.
            
        Returns:
            Dict {expert_id: ExpertOutput}.
        """
        outputs = {}
        
        for expert_id in expert_ids:
            expert = self._registry.get(expert_id)
            if expert is None:
                continue
            
            if not expert.can_handle(window):
                continue
            
            try:
                output = expert.predict(window)
                outputs[expert_id] = output
            except Exception:
                # Skip failed experts (fail-silent)
                continue
        
        return outputs
    
    def _enrich_prediction(
        self,
        prediction: Prediction,
        moe_metadata: MoEMetadata,
        window: SensorWindow
    ) -> Prediction:
        """Enriquece predicción con metadata MoE.
        
        Args:
            prediction: Predicción base de fusión.
            moe_metadata: Metadata de ejecución MoE.
            window: Ventana original.
            
        Returns:
            Prediction enriquecida.
        """
        # Crear nueva prediction con metadata enriquecida
        enriched_metadata = {
            **prediction.metadata,
            "moe": {
                "enabled": moe_metadata.moe_enabled,
                "selected_experts": moe_metadata.selected_experts,
                "sparsity_k": moe_metadata.sparsity_k,
                "dominant_expert": moe_metadata.dominant_expert,
                "gating_probs": moe_metadata.gating_probs,
                "fusion_weights": moe_metadata.fusion_weights,
                "latency_ms": round(moe_metadata.total_latency_ms, 2),
            },
        }
        
        return Prediction(
            series_id=str(window.sensor_id) if hasattr(window, 'sensor_id') else "unknown",
            predicted_value=prediction.predicted_value,
            confidence=prediction.confidence,
            trend=prediction.trend,
            timestamp=prediction.timestamp,
            metadata=enriched_metadata,
        )
    
    def can_handle(self, n_points: int) -> bool:
        """Verifica si puede operar con n_points datos.
        
        Delega al fallback si MoE deshabilitado.
        
        Args:
            n_points: Número de puntos disponibles.
            
        Returns:
            True si puede generar predicción.
        """
        if not self._moe_enabled:
            return self._fallback.can_handle(n_points)
        
        # Verificar si al menos un experto puede manejar los puntos
        candidates = self._registry.get_candidates(
            ContextVector(regime="stable", domain="iot", n_points=n_points, signal_features={})
        )
        return len(candidates) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del gateway.
        
        Returns:
            Dict con métricas de operación.
        """
        return {
            "name": self.name,
            "moe_enabled": self._moe_enabled,
            "sparsity_k": self._sparsity_k,
            "expert_pool_size": len(self._registry),
            "gating_type": self._gating.__class__.__name__,
        }
