"""Experto asimétrico de invariantes en reposo (Escala 10X).

SRP: Monitorea la envolvente estática y límites extremos en la escala más comprimida.
Costo computacional mínimo O(1) / O(N_comprimido).
Afinidad declarada: RepresentationLevel.TEN_X.
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


class RestingInvariantExpert(AsymmetricExpertPort):
    """Experto O(1) de envolvente para regímenes estables y reposo a 10X."""

    affinity: RepresentationLevel = RepresentationLevel.TEN_X

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        margin: float | None = None,
        name: str = "resting_invariants_10x",
        compute_cost_estimate: float = 0.05,
    ) -> None:
        self.name = name
        self.affinity = RepresentationLevel.TEN_X
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.margin = margin if margin is not None and margin > 1e-6 else max(1.0, (upper_bound - lower_bound) * 0.1)
        self.compute_cost_estimate = compute_cost_estimate

    def evaluate(self, series_slice: Sequence[float]) -> EvidenceScore:
        """Calcula la evidencia de anomalía comprobando violación de la envolvente de reposo."""
        if not series_slice:
            return EvidenceScore(
                expert_name=self.name,
                representation_affinity=self.affinity,
                anomaly_probability=0.01,
                compute_cost_estimate=self.compute_cost_estimate,
                metadata={"reason": "empty_slice"},
            )

        min_val = min(series_slice)
        max_val = max(series_slice)

        lower_violation = max(0.0, self.lower_bound - min_val)
        upper_violation = max(0.0, max_val - self.upper_bound)
        excess = max(lower_violation, upper_violation)

        if excess <= 1e-9:
            # Envolvente respetada al 100%: certeza de operación nominal en reposo
            prob = 0.01
        else:
            # Infracción de envolvente: probabilidad sigmoidal/exponencial creciente
            prob = 1.0 - math.exp(-excess / self.margin)
            prob = max(0.01, min(0.999, prob))

        return EvidenceScore(
            expert_name=self.name,
            representation_affinity=self.affinity,
            anomaly_probability=round(prob, 4),
            compute_cost_estimate=self.compute_cost_estimate,
            metadata={
                "min_val": round(min_val, 4),
                "max_val": round(max_val, 4),
                "excess": round(excess, 4),
            },
        )
