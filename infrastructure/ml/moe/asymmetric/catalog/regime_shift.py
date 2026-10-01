"""Experto asimétrico de cambio de régimen y deriva persistente (Escala 2X).

SRP: Monitorea el desplazamiento del centroide y tendencias graduales a resolución intermedia.
Especializado en detectar derivas sutiles (como el Evento 3 de NAB) sin pagar el costo de RAW.
Afinidad declarada: RepresentationLevel.TWO_X.
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


class RegimeShiftExpert(AsymmetricExpertPort):
    """Experto de deriva estructural y desplazamiento de media a escala 2X."""

    affinity: RepresentationLevel = RepresentationLevel.TWO_X

    def __init__(
        self,
        baseline_center: float,
        baseline_spread: float,
        drift_sensitivity: float = 1.8,
        name: str = "regime_shift_2x",
        compute_cost_estimate: float = 0.20,
    ) -> None:
        self.name = name
        self.affinity = RepresentationLevel.TWO_X
        self.baseline_center = baseline_center
        self.baseline_spread = max(1e-6, baseline_spread)
        self.drift_sensitivity = drift_sensitivity
        self.compute_cost_estimate = compute_cost_estimate

    def evaluate(self, series_slice: Sequence[float]) -> EvidenceScore:
        """Calcula la evidencia de anomalía evaluando desplazamiento de régimen en el slice."""
        if not series_slice:
            return EvidenceScore(
                expert_name=self.name,
                representation_affinity=self.affinity,
                anomaly_probability=0.01,
                compute_cost_estimate=self.compute_cost_estimate,
                metadata={"reason": "empty_slice"},
            )

        n = len(series_slice)
        slice_mean = sum(series_slice) / n
        deviation = abs(slice_mean - self.baseline_center)
        z_score = deviation / self.baseline_spread

        # Detección de pendiente/tendencia si hay al menos 3 puntos
        slope = 0.0
        if n >= 3:
            # Regresión lineal simple sobre índices normalizados
            t_mean = (n - 1) / 2.0
            var_t = sum((i - t_mean) ** 2 for i in range(n))
            if var_t > 1e-9:
                cov_t_x = sum((i - t_mean) * (series_slice[i] - slice_mean) for i in range(n))
                slope = cov_t_x / var_t

        # Ajuste de z_score según consistencia de pendiente
        normalized_slope = abs(slope) * n / self.baseline_spread
        effective_z = z_score + 0.5 * normalized_slope

        # Probabilidad logística calibrada centrada en drift_sensitivity
        # Cuando effective_z == drift_sensitivity, prob = 0.5
        prob = 1.0 / (1.0 + math.exp(-1.5 * (effective_z - self.drift_sensitivity)))
        prob = max(0.005, min(0.995, prob))

        return EvidenceScore(
            expert_name=self.name,
            representation_affinity=self.affinity,
            anomaly_probability=round(prob, 4),
            compute_cost_estimate=self.compute_cost_estimate,
            metadata={
                "slice_mean": round(slice_mean, 4),
                "z_score": round(z_score, 4),
                "slope": round(slope, 6),
                "effective_z": round(effective_z, 4),
            },
        )
