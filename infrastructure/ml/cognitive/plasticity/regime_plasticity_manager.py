"""Gestor de plasticidad bayesiana adaptativa y circuit-breakers por régimen."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from domain.entities.cognitive.post_mortem_evaluator import (
    PostMortemRecord,
)


class RegimePlasticityManager:
    """Ajusta pesos de expertos por régimen con inhibición temporal ante fallos repetidos."""

    def __init__(
        self,
        expert_names: List[str],
        learning_rate: float = 0.15,
        min_weight: float = 0.05,
        max_consecutive_failures: int = 4,
    ) -> None:
        self._expert_names = list(expert_names)
        self._eta = learning_rate
        self._min_weight = min_weight
        self._max_failures = max_consecutive_failures

        # Por cada régimen: {expert: weight}
        self._weights: Dict[str, Dict[str, float]] = {}
        # Contador de fallos consecutivos: {regime: {expert: count}}
        self._consecutive_failures: Dict[str, Dict[str, int]] = {}

    def _init_regime_if_missing(self, regime: str) -> None:
        if regime not in self._weights:
            n = len(self._expert_names)
            base_w = 1.0 / n if n > 0 else 1.0
            self._weights[regime] = {exp: base_w for exp in self._expert_names}
            self._consecutive_failures[regime] = {exp: 0 for exp in self._expert_names}

    def get_weights(self, regime: str = "default") -> Dict[str, float]:
        """Retorna los pesos normalizados actuales para el régimen."""
        self._init_regime_if_missing(regime)
        return dict(self._weights[regime])

    def is_inhibited(self, expert_name: str, regime: str = "default") -> bool:
        """Determina si un experto disparó su circuit-breaker en este régimen."""
        self._init_regime_if_missing(regime)
        return self._consecutive_failures[regime].get(expert_name, 0) >= self._max_failures

    def update_from_post_mortem(self, record: PostMortemRecord) -> None:
        """Actualiza pesos e inhibición con base en la evaluación posterior auditada."""
        regime = record.regime
        self._init_regime_if_missing(regime)

        weights = self._weights[regime]
        failures = self._consecutive_failures[regime]

        # 1. Actualizar contadores de fallos y pérdidas
        for exp in self._expert_names:
            is_correct = record.directional_correctness.get(exp, False)
            if is_correct:
                failures[exp] = max(0, failures[exp] - 1)
            else:
                failures[exp] = failures.get(exp, 0) + 1

            # Actualización exponencial por pérdida multiplicativa
            err = record.expert_errors.get(exp, 1.0)
            if failures[exp] >= self._max_failures:
                # Circuit breaker: peso nulo temporal
                weights[exp] = 0.0
            else:
                # Modulación exponencial: w_i <- w_i * exp(-η * err)
                penalty = math.exp(-self._eta * min(err, 10.0))
                weights[exp] = max(self._min_weight, weights[exp] * penalty)

        # 2. Normalizar a simplex (sum(w) = 1.0)
        total_w = sum(weights.values())
        if total_w > 1e-9:
            for exp in weights:
                weights[exp] = weights[exp] / total_w
        else:
            # Si todos se inhibieron, reset equilibrado
            n = len(self._expert_names)
            for exp in weights:
                weights[exp] = 1.0 / n

    def export_state(self) -> Dict[str, Any]:
        """Serializa el estado para persistencia segura."""
        return {
            "schema_version": 1,
            "expert_names": self._expert_names,
            "weights": self._weights,
            "consecutive_failures": self._consecutive_failures,
        }

    def import_state(self, payload: Dict[str, Any]) -> None:
        """Restaura el estado aprendido."""
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Payload de plasticidad inválido o schema_version desconocido")
        self._weights = payload.get("weights", {})
        self._consecutive_failures = payload.get("consecutive_failures", {})
