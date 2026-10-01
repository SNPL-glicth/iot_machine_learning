"""Experto asimétrico de alta frecuencia y choque instantáneo (Escala RAW).

SRP: Monitorea innovaciones diferenciales de alta frecuencia y picos abruptos.
Especializado en transitorios rápidos, choques e impulsos instantáneos.
Afinidad declarada: RepresentationLevel.RAW.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from iot_machine_learning.domain.entities.representation_evidence import (
    EvidenceScore,
    RepresentationLevel,
)
from iot_machine_learning.domain.ports.asymmetric_expert_port import (
    AsymmetricExpertPort,
)


class HighFrequencyExpert(AsymmetricExpertPort):
    """Experto de innovación diferencial y picos en resolución completa (RAW)."""

    affinity: RepresentationLevel = RepresentationLevel.RAW

    def __init__(
        self,
        shock_threshold: float,
        shock_sensitivity: float = 1.5,
        name: str = "high_frequency_raw",
        compute_cost_estimate: float = 1.0,
    ) -> None:
        self.name = name
        self.affinity = RepresentationLevel.RAW
        self.shock_threshold = max(1e-6, shock_threshold)
        self.shock_sensitivity = shock_sensitivity
        self.compute_cost_estimate = compute_cost_estimate

    def evaluate(self, series_slice: Sequence[float]) -> EvidenceScore:
        """Calcula la evidencia de choque evaluando primeras diferencias e innovaciones puntuales."""
        if not series_slice or len(series_slice) < 2:
            return EvidenceScore(
                expert_name=self.name,
                representation_affinity=self.affinity,
                anomaly_probability=0.01,
                compute_cost_estimate=self.compute_cost_estimate,
                metadata={"reason": "insufficient_points_for_diff"},
            )

        # Calcular primeras diferencias continuas
        diffs = [abs(series_slice[i] - series_slice[i - 1]) for i in range(1, len(series_slice))]
        max_diff = max(diffs)
        mean_diff = sum(diffs) / len(diffs)

        z_shock = max_diff / self.shock_threshold

        # Probabilidad logística calibrada para choques
        prob = 1.0 / (1.0 + math.exp(-2.0 * (z_shock - self.shock_sensitivity)))
        prob = max(0.005, min(0.999, prob))

        return EvidenceScore(
            expert_name=self.name,
            representation_affinity=self.affinity,
            anomaly_probability=round(prob, 4),
            compute_cost_estimate=self.compute_cost_estimate,
            metadata={
                "max_diff": round(max_diff, 4),
                "mean_diff": round(mean_diff, 4),
                "z_shock": round(z_shock, 4),
            },
        )
