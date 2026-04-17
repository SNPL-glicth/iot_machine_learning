"""SparseFusionLayer — fusión ponderada de k expertos seleccionados.

SRP: Solo fusión, no gating, no dispatch, no routing.
Toma outputs de k expertos y produce una Prediction del dominio.

ISO 42001: Registra pesos usados para trazabilidad de decisiones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import math

from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.ports.expert_port import ExpertOutput


@dataclass(frozen=True)
class FusionWeights:
    """Pesos normalizados resultantes de la fusión.
    
    Attributes:
        normalized: Dict {expert_id: peso_normalizado}.
        dominant: ID del experto con mayor peso.
        sparsity_k: Número de expertos fusionados.
    """
    normalized: Dict[str, float]
    dominant: str
    sparsity_k: int


class SparseFusionLayer:
    """Capa de fusión sparse para arquitectura MoE.
    
    Responsabilidad única (SRP): Fusionar outputs de k expertos.
    
    Algoritmo:
    1. Normalizar pesos a suma 1.0
    2. Media ponderada de predicciones
    3. Media ponderada de confianzas
    4. Tendencia del experto dominante
    
    Args:
        min_confidence: Mínimo para considerar output válido.
    """
    
    def __init__(self, min_confidence: float = 0.0):
        """Inicializa capa de fusión.
        
        Args:
            min_confidence: Umbral mínimo de confianza.
        """
        self._min_confidence = min_confidence
    
    def fuse(
        self,
        expert_outputs: Dict[str, ExpertOutput],
        weights: Dict[str, float],
    ) -> Prediction:
        """Fusiona outputs de k expertos en una predicción.
        
        Args:
            expert_outputs: Dict {expert_id: ExpertOutput} de k expertos.
            weights: Pesos de gating (pueden no estar normalizados).
            
        Returns:
            Prediction del dominio fusionada.
            
        Raises:
            ValueError: Si no hay outputs o weights vacíos.
        """
        if not expert_outputs:
            raise ValueError("No hay expert_outputs para fusionar")
        if not weights:
            raise ValueError("No hay weights para fusionar")
        
        # 1. Normalizar pesos
        normalized = self._normalize_weights(weights)
        
        # 2. Calcular predicción ponderada
        fused_value = self._weighted_prediction(expert_outputs, normalized)
        
        # 3. Calcular confianza ponderada
        fused_confidence = self._weighted_confidence(expert_outputs, normalized)
        
        # 4. Determinar tendencia del experto dominante
        dominant = max(normalized.items(), key=lambda x: x[1])[0]
        fused_trend = expert_outputs[dominant].trend
        
        # 5. Calcular incertidumbre (desviación entre predicciones)
        uncertainty = self._compute_uncertainty(expert_outputs, fused_value)
        
        # 6. Construir metadata
        metadata = {
            "fusion": {
                "sparsity_k": len(expert_outputs),
                "weights_used": normalized,
                "dominant_expert": dominant,
                "uncertainty": round(uncertainty, 6),
            }
        }
        
        # 7. Crear Prediction del dominio
        return Prediction(
            series_id="fused",  # Placeholder, se actualiza en gateway
            predicted_value=fused_value,
            confidence_score=fused_confidence,
            trend=fused_trend,
            engine_name="moe_fusion",
            metadata=metadata,
        )
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normaliza pesos para sumar 1.0.
        
        Args:
            weights: Pesos (posiblemente no normalizados).
            
        Returns:
            Pesos normalizados.
        """
        total = sum(weights.values())
        if total < 1e-9:
            # Fallback a uniforme
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        
        return {k: v / total for k, v in weights.items()}
    
    def _weighted_prediction(
        self,
        outputs: Dict[str, ExpertOutput],
        weights: Dict[str, float]
    ) -> float:
        """Calcula media ponderada de predicciones.
        
        Args:
            outputs: Outputs de expertos.
            weights: Pesos normalizados.
            
        Returns:
            Valor predicho fusionado.
        """
        return sum(
            outputs[eid].prediction * weights[eid]
            for eid in outputs
        )
    
    def _weighted_confidence(
        self,
        outputs: Dict[str, ExpertOutput],
        weights: Dict[str, float]
    ) -> float:
        """Calcula media ponderada de confianzas.
        
        Args:
            outputs: Outputs de expertos.
            weights: Pesos normalizados.
            
        Returns:
            Confianza fusionada.
        """
        return sum(
            outputs[eid].confidence * weights[eid]
            for eid in outputs
        )
    
    def _compute_uncertainty(
        self,
        outputs: Dict[str, ExpertOutput],
        mean_prediction: float
    ) -> float:
        """Calcula incertidumbre como desviación estándar ponderada.
        
        Args:
            outputs: Outputs de expertos.
            mean_prediction: Predicción media (fusionada).
            
        Returns:
            Incertidumbre (std dev de predicciones).
        """
        if len(outputs) < 2:
            return 0.0
        
        # Varianza de predicciones
        predictions = [o.prediction for o in outputs.values()]
        variance = sum(
            (p - mean_prediction) ** 2
            for p in predictions
        ) / len(predictions)
        
        return math.sqrt(variance)
    
    def get_fusion_weights(
        self,
        weights: Dict[str, float]
    ) -> FusionWeights:
        """Obtiene pesos normalizados sin fusionar.
        
        Utilidad para inspección/debugging.
        
        Args:
            weights: Pesos de gating.
            
        Returns:
            FusionWeights con metadatos.
        """
        normalized = self._normalize_weights(weights)
        dominant = max(normalized.items(), key=lambda x: x[1])[0]
        
        return FusionWeights(
            normalized=normalized,
            dominant=dominant,
            sparsity_k=len(weights),
        )
