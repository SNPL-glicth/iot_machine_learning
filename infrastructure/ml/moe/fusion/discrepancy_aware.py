"""DiscrepancyAwareFusion — fusión que penaliza discrepancia entre expertos.

Reemplaza SparseFusionLayer:
- Si std de predicciones > threshold, penaliza confianza fusionada.
- La confianza fusionada NUNCA supera max(expert_confidences) bajo alta discrepancia.
- Threshold configurable (default = 2.0).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.ports.expert_port import ExpertOutput

from .sparse_fusion import FusionWeights


class DiscrepancyAwareFusion:
    """Capa de fusión consciente de discrepancia entre expertos.

    Responsabilidad única (SRP): fusionar outputs penalizando divergencia.

    Args:
        discrepancy_threshold: Umbral de std sobre el cual se penaliza.
            Default = 2.0 (unidades absolutas de la señal).
        min_confidence: Mínimo para considerar output válido.
    """

    def __init__(
        self,
        discrepancy_threshold: float = 2.0,
        min_confidence: float = 0.0,
    ) -> None:
        self._discrepancy_threshold = discrepancy_threshold
        self._min_confidence = min_confidence

    def fuse(
        self,
        expert_outputs: Dict[str, ExpertOutput],
        weights: Dict[str, float],
    ) -> Prediction:
        """Fusiona outputs de expertos en una Prediction del dominio.

        Args:
            expert_outputs: Dict {expert_id: ExpertOutput}.
            weights: Pesos de gating (pueden no estar normalizados).

        Returns:
            Prediction fusionada.

        Raises:
            ValueError: Si no hay outputs o weights vacíos.
        """
        if not expert_outputs:
            raise ValueError("No hay expert_outputs para fusionar")
        if not weights:
            raise ValueError("No hay weights para fusionar")

        normalized = self._normalize_weights(weights)

        # Filtrar por min_confidence
        valid_outputs = {
            eid: out
            for eid, out in expert_outputs.items()
            if out.confidence >= self._min_confidence
        }
        if not valid_outputs:
            valid_outputs = expert_outputs  # fallback: usar todos

        # Re-normalizar sobre los válidos
        valid_weights = {eid: normalized.get(eid, 0.0) for eid in valid_outputs}
        valid_weights = self._normalize_weights(valid_weights)

        fused_value = self._weighted_prediction(valid_outputs, valid_weights)
        fused_confidence = self._weighted_confidence(valid_outputs, valid_weights)
        dominant = max(valid_weights.items(), key=lambda x: x[1])[0]
        fused_trend = valid_outputs[dominant].trend

        # Penalización por discrepancia
        std_pred = self._std_of_predictions(valid_outputs)
        max_conf = max(o.confidence for o in valid_outputs.values())

        if std_pred > self._discrepancy_threshold:
            # Alta discrepancia: cap + penalización proporcional
            fused_confidence = min(fused_confidence, max_conf)
            penalty = max(0.5, self._discrepancy_threshold / max(std_pred, 1e-9))
            fused_confidence *= penalty

        uncertainty = std_pred

        metadata = {
            "fusion": {
                "sparsity_k": len(valid_outputs),
                "weights_used": valid_weights,
                "dominant_expert": dominant,
                "uncertainty": round(uncertainty, 6),
                "std_predictions": round(std_pred, 6),
                "discrepancy_penalized": std_pred > self._discrepancy_threshold,
            }
        }

        return Prediction(
            series_id="fused",
            predicted_value=fused_value,
            confidence_score=fused_confidence,
            trend=fused_trend,
            engine_name="moe_discrepancy_fusion",
            metadata=metadata,
        )

    def get_fusion_weights(
        self, weights: Dict[str, float]
    ) -> FusionWeights:
        """Obtiene pesos normalizados sin fusionar."""
        normalized = self._normalize_weights(weights)
        dominant = max(normalized.items(), key=lambda x: x[1])[0]
        return FusionWeights(
            normalized=normalized,
            dominant=dominant,
            sparsity_k=len(weights),
        )

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total < 1e-9:
            n = len(weights)
            return {k: 1.0 / n for k in weights} if n > 0 else {}
        return {k: v / total for k, v in weights.items()}

    @staticmethod
    def _weighted_prediction(
        outputs: Dict[str, ExpertOutput],
        weights: Dict[str, float],
    ) -> float:
        return sum(
            outputs[eid].prediction * weights[eid]
            for eid in outputs
        )

    @staticmethod
    def _weighted_confidence(
        outputs: Dict[str, ExpertOutput],
        weights: Dict[str, float],
    ) -> float:
        return sum(
            outputs[eid].confidence * weights[eid]
            for eid in outputs
        )

    @staticmethod
    def _std_of_predictions(outputs: Dict[str, ExpertOutput]) -> float:
        """Std de las predicciones de expertos."""
        if len(outputs) < 2:
            return 0.0
        predictions = [o.prediction for o in outputs.values()]
        mean = sum(predictions) / len(predictions)
        variance = sum((p - mean) ** 2 for p in predictions) / len(predictions)
        return math.sqrt(variance)
