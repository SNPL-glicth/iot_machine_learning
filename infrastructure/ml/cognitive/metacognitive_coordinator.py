"""Coordinador central del bucle metacognitivo y meta-aprendizaje."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from domain.entities.cognitive.metacognitive_tracker import (
    MetacognitiveStatus,
    MetacognitiveTracker,
)
from domain.entities.cognitive.post_mortem_evaluator import (
    PostMortemEvaluator,
    PostMortemRecord,
)
from infrastructure.ml.cognitive.plasticity.regime_plasticity_manager import (
    RegimePlasticityManager,
)


class MetacognitiveCoordinator:
    """Orquesta el ciclo completo de auto-evaluación, diagnóstico y plasticidad."""

    def __init__(
        self,
        expert_names: List[str],
        horizon_steps: int = 11,
        learning_rate: float = 0.15,
        window_size: int = 20,
    ) -> None:
        self._expert_names = list(expert_names)
        self._default_horizon = horizon_steps
        self._evaluator = PostMortemEvaluator()
        self._tracker = MetacognitiveTracker(window_size=window_size)
        self._plasticity = RegimePlasticityManager(
            expert_names=expert_names, learning_rate=learning_rate
        )

    @property
    def plasticity(self) -> RegimePlasticityManager:
        return self._plasticity

    @property
    def tracker(self) -> MetacognitiveTracker:
        return self._tracker

    def register_inference(
        self,
        prediction_id: str,
        prev_value: float,
        predicted_value: float,
        expert_predictions: Dict[str, float],
        regime: str = "default",
        confidence: float = 0.5,
        lambda_t: float = 0.0,
        expert_variance: float = 0.0,
        horizon_steps: Optional[int] = None,
    ) -> None:
        """Registra la emisión de una inferencia para auditoría diferida."""
        h = horizon_steps if horizon_steps is not None else self._default_horizon
        self._evaluator.record_prediction(
            prediction_id=prediction_id,
            prev_value=prev_value,
            predicted_value=predicted_value,
            expert_predictions=expert_predictions,
            regime=regime,
            confidence=confidence,
            lambda_t=lambda_t,
            expert_variance=expert_variance,
            horizon_steps=h,
        )

    def process_step(
        self, current_value: float, is_outlier: bool = False
    ) -> List[PostMortemRecord]:
        """Avanza el reloj temporal, procesa predicciones maduras y adapta el sistema."""
        matured_records = self._evaluator.step(current_value, is_outlier=is_outlier)
        for record in matured_records:
            self._tracker.record_outcome(record)
            self._plasticity.update_from_post_mortem(record)
        return matured_records

    def modulate_exploration_factor(self, base_lambda: float, regime: str = "default") -> float:
        """Ajusta lambda_t en función de la meta-competencia en el régimen."""
        return self._tracker.modulate_exploration_factor(base_lambda, regime)

    def get_expert_weights(self, regime: str = "default") -> Dict[str, float]:
        """Obtiene la distribución adaptativa de pesos de los expertos."""
        return self._plasticity.get_weights(regime)

    def get_metacognitive_status(self, regime: str = "default") -> MetacognitiveStatus:
        """Obtiene el diagnóstico de comprensión del régimen actual."""
        return self._tracker.get_status(regime)

    def export_state(self) -> Dict[str, Any]:
        """Serializa el estado para snapshots."""
        return {
            "schema_version": 1,
            "expert_names": self._expert_names,
            "plasticity": self._plasticity.export_state(),
        }

    def import_state(self, payload: Dict[str, Any]) -> None:
        """Restaura el estado aprendido."""
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Payload de coordinador metacognitivo inválido")
        if "plasticity" in payload:
            self._plasticity.import_state(payload["plasticity"])
