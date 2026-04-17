"""PredictionEnricher — Enriquecimiento de predicciones con metadata MoE.

Extraído de MoEGateway como servicio independiente siguiendo SRP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow


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


class PredictionEnricher:
    """Servicio de enriquecimiento de predicciones con metadata MoE.
    
    Responsabilidad única: agregar metadata de ejecución MoE a predicciones.
    """
    
    def enrich(
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
            Prediction enriquecida con metadata MoE en prediction.metadata["moe"].
        """
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
            confidence_score=prediction.confidence_score,
            trend=prediction.trend,
            engine_name=prediction.metadata.get("engine_name", "moe_fusion"),
            metadata=enriched_metadata,
        )
    
    def get_stats(
        self,
        moe_enabled: bool,
        sparsity_k: int,
        registry_size: int,
        gating_type: str,
    ) -> Dict[str, Any]:
        """Genera estadísticas del gateway.
        
        Args:
            moe_enabled: Si MoE está habilitado.
            sparsity_k: Número de expertos top-k.
            registry_size: Tamaño del pool de expertos.
            gating_type: Nombre de la clase de gating.
            
        Returns:
            Dict con métricas de operación.
        """
        return {
            "name": "moe_gateway",
            "moe_enabled": moe_enabled,
            "sparsity_k": sparsity_k,
            "expert_pool_size": registry_size,
            "gating_type": gating_type,
        }
