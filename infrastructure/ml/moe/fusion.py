"""SparseFusionLayer — fusión ponderada de k expertos (no todos).

Diferencia clave con WeightedFusion: solo fusiona k expertos seleccionados,
no todos los disponibles. Esto es O(k) vs O(N).

ISO 42001: Registra contribuciones de cada experto para trazabilidad.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from iot_machine_learning.domain.ports.expert_port import ExpertOutput


@dataclass(frozen=True)
class FusedResult:
    """Resultado de fusión de múltiples expertos.
    
    Contiene valor fusionado, métricas de calidad y metadata para
    explicabilidad.
    
    Attributes:
        value: Valor predicho final (fusionado).
        confidence: Confianza agregada [0.0, 1.0].
        trend: Tendencia mayoritaria ("up", "down", "stable").
        weights: Pesos normalizados usados por cada experto.
        uncertainty: Varianza entre predicciones (divergencia).
        selected_expert: Experto con mayor peso.
        computation_saved_pct: Porcentaje de computación ahorrada.
    """
    value: float
    confidence: float
    trend: str
    weights: Dict[str, float]
    uncertainty: float
    selected_expert: str
    computation_saved_pct: float
    
    def __post_init__(self):
        if self.trend not in ("up", "down", "stable"):
            raise ValueError(f"trend inválido: {self.trend}")


class SparseFusionLayer:
    """Fusión ponderada de k expertos seleccionados.
    
    Implementa fusión "sparse": solo considera k expertos activados
    por el gating network, no todos los N disponibles.
    
    Algoritmo:
    1. Renormalizar pesos de los k expertos a suma 1.0
    2. Media ponderada de predicciones
    3. Media ponderada de confianzas
    4. Voto mayoritario ponderado para tendencia
    5. Calcular incertidumbre (varianza ponderada)
    
    Args:
        min_confidence_threshold: Mínimo para considerar predicción válida.
        max_disagreement_threshold: Máxima divergencia permitida antes de alertar.
    """
    
    def __init__(
        self,
        min_confidence_threshold: float = 0.3,
        max_disagreement_threshold: float = 5.0,
    ):
        self._min_confidence = min_confidence_threshold
        self._max_disagreement = max_disagreement_threshold
    
    def fuse(
        self,
        outputs: Dict[str, ExpertOutput],
        gating_probs: Dict[str, float],
        total_experts_available: int,
    ) -> FusedResult:
        """Fusiona salidas de expertos seleccionados.
        
        Args:
            outputs: Dict {expert_id: ExpertOutput} de los k expertos ejecutados.
            gating_probs: Probabilidades del gating (incluye todos, no solo k).
            total_experts_available: N total de expertos en el pool.
            
        Returns:
            FusedResult con predicción fusionada y métricas.
            
        Raises:
            ValueError: Si outputs vacío o gating_probs inconsistente.
        """
        if not outputs:
            raise ValueError("No hay outputs para fusionar")
        
        # Filtrar solo los k expertos que fueron ejecutados
        selected_ids = list(outputs.keys())
        k = len(selected_ids)
        
        # Extraer probabilidades de los k expertos seleccionados
        selected_probs = {eid: gating_probs.get(eid, 0.0) for eid in selected_ids}
        
        # Renormalizar pesos para que sumen 1.0
        weights = self._normalize_weights(selected_probs)
        
        # Media ponderada de predicciones
        fused_value = sum(
            outputs[eid].prediction * weights[eid]
            for eid in selected_ids
        )
        
        # Media ponderada de confianzas
        fused_confidence = sum(
            outputs[eid].confidence * weights[eid]
            for eid in selected_ids
        )
        
        # Voto mayoritario ponderado para tendencia
        trend_votes: Dict[str, float] = {"up": 0.0, "down": 0.0, "stable": 0.0}
        for eid in selected_ids:
            trend_votes[outputs[eid].trend] += weights[eid]
        
        fused_trend = max(trend_votes.items(), key=lambda x: x[1])[0]
        
        # Calcular incertidumbre (varianza ponderada entre predicciones)
        mean_pred = sum(o.prediction for o in outputs.values()) / k
        variance = sum(
            weights[eid] * (outputs[eid].prediction - mean_pred) ** 2
            for eid in selected_ids
        )
        uncertainty = math.sqrt(variance)
        
        # Experto con mayor peso
        selected_expert = max(weights.items(), key=lambda x: x[1])[0]
        
        # Porcentaje de computación ahorrada
        saved_pct = (1 - k / total_experts_available) * 100 if total_experts_available > 0 else 0.0
        
        return FusedResult(
            value=fused_value,
            confidence=fused_confidence,
            trend=fused_trend,
            weights=weights,
            uncertainty=uncertainty,
            selected_expert=selected_expert,
            computation_saved_pct=saved_pct,
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
    
    def compute_expert_disagreement(
        self,
        outputs: Dict[str, ExpertOutput],
        weights: Dict[str, float]
    ) -> Tuple[float, bool]:
        """Computa nivel de desacuerdo entre expertos.
        
        Args:
            outputs: Salidas de expertos.
            weights: Pesos usados.
            
        Returns:
            Tuple (disagreement_score, is_significant).
            disagreement_score: 0.0 (consenso) a 1.0 (máxima divergencia).
            is_significant: True si supera threshold configurado.
        """
        if len(outputs) < 2:
            return 0.0, False
        
        # Calcular rango ponderado de predicciones
        predictions = [o.prediction for o in outputs.values()]
        pred_range = max(predictions) - min(predictions)
        
        # Normalizar por media (coeficiente de variación efectivo)
        mean_pred = sum(p * weights.get(eid, 1.0 / len(outputs)) 
                       for eid, p in zip(outputs.keys(), predictions))
        
        if abs(mean_pred) < 1e-9:
            return 0.0, False
        
        disagreement = pred_range / abs(mean_pred)
        is_significant = disagreement > self._max_disagreement
        
        # Normalizar a [0, 1] usando sigmoide suave
        normalized_disagreement = disagreement / (1 + disagreement)
        
        return normalized_disagreement, is_significant
    
    def should_alert_divergence(
        self,
        outputs: Dict[str, ExpertOutput],
        threshold_pct: float = 10.0
    ) -> Tuple[bool, str]:
        """Determina si expertos divergen significativamente.
        
        Args:
            outputs: Salidas de expertos.
            threshold_pct: Umbral de divergencia en porcentaje.
            
        Returns:
            Tuple (should_alert, explanation).
        """
        if len(outputs) < 2:
            return False, "Solo un experto, no hay divergencia"
        
        predictions = [o.prediction for o in outputs.values()]
        mean_pred = sum(predictions) / len(predictions)
        
        if abs(mean_pred) < 1e-9:
            return False, "Media cercana a cero, divergencia no aplicable"
        
        max_deviation = max(abs(p - mean_pred) for p in predictions)
        deviation_pct = (max_deviation / abs(mean_pred)) * 100
        
        if deviation_pct > threshold_pct:
            return True, f"Divergencia {deviation_pct:.1f}% supera umbral {threshold_pct}%"
        
        return False, f"Divergencia {deviation_pct:.1f}% dentro de tolerancia"
