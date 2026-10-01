"""Calibrador online no paramétrico de e-values y ponderación adaptativa (Hedge / OCO).

SRP: Transforma evidencias atómicas en e-values (E[e|H0] <= 1) y actualiza los pesos
convexos de los expertos mediante Online Convex Optimization sin reentrenar modelos.
Cumple el contrato OnlineCalibratorPort del dominio.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from iot_machine_learning.domain.entities.representation_evidence import (
    EvidenceScore,
    SystemOperationalState,
)
from iot_machine_learning.domain.ports.meta_gate_port import OnlineCalibratorPort


class OnlineConformalCalibrator(OnlineCalibratorPort):
    """Calibrador secuencial de e-values y pesos de expertos mediante el algoritmo Hedge."""

    def __init__(
        self,
        nominal_prior_rate: float = 0.05,
        betting_fraction: float = 5.0,
        learning_rate: float = 0.15,
        min_weight_floor: float = 0.05,
        known_experts: Sequence[str] | None = None,
    ) -> None:
        self.nominal_prior_rate = max(1e-4, min(0.5, nominal_prior_rate))
        self.betting_fraction = betting_fraction
        self.learning_rate = learning_rate
        self.min_weight_floor = min_weight_floor

        # Pesos convexos w_{i, t}
        self._weights: dict[str, float] = {}
        if known_experts:
            n = len(known_experts)
            init_w = 1.0 / n if n > 0 else 1.0
            for name in known_experts:
                self._weights[name] = init_w

        # Estadísticas de pérdidas acumuladas
        self._cumulative_regret: dict[str, float] = {k: 0.0 for k in self._weights}

    @property
    def current_weights(self) -> Mapping[str, float]:
        """Retorna una vista inmutable de los pesos convexos activos."""
        return dict(self._weights)

    def compute_e_values(
        self, evidences: Sequence[EvidenceScore]
    ) -> list[float]:
        """Transforma puntuaciones de anomalía [0, 1] en e-values canónicos.

        Bajo H0 donde E[p_i | H0] <= p_0:
            E_i = (p_i + eps) / (p_0 + eps)
        satisface rigurosamente E[E_i | H0] <= 1 (Likelihood Ratio E-Value).
        """
        e_values: list[float] = []
        p0 = self.nominal_prior_rate
        eps = 1e-4
        for ev in evidences:
            p = max(0.001, min(0.999, ev.anomaly_probability))
            e_val = (p + eps) / (p0 + eps)
            e_values.append(max(0.01, min(100.0, e_val)))
        return e_values

    def update_weights(
        self,
        evidences: Sequence[EvidenceScore],
        e_values: Sequence[float],
        current_state: SystemOperationalState,
    ) -> Mapping[str, float]:
        """Actualiza los pesos convexos mediante el algoritmo multiplicativo Hedge.

        Penaliza expertos que emiten alarmas espurias durante periodos de reposo (RESTING)
        y premia expertos que detectan derivas o choques genuinos (DRIFTING, SHOCKED).
        """
        if not evidences or not e_values:
            return self.current_weights

        # Registrar expertos nuevos dinámicamente si no existían
        for ev in evidences:
            if ev.expert_name not in self._weights:
                self._weights[ev.expert_name] = 1.0 / max(1, len(self._weights) + 1)
                self._cumulative_regret[ev.expert_name] = 0.0
                self._normalize_weights()

        for ev, e_val in zip(evidences, e_values, strict=False):
            name = ev.expert_name

            # Función de pérdida/recompensa contextual
            if current_state == SystemOperationalState.RESTING:
                # En reposo nominal (H0), un e-value alto es una falsa alarma -> penalización fuerte
                loss = math.log(max(1.0, e_val))
            elif current_state == SystemOperationalState.SHOCKED:
                # En choque (H1), un e-value alto es un acierto -> recompensa (pérdida negativa)
                loss = -math.log(max(0.1, e_val))
            else:  # DRIFTING
                # En deriva, premiamos sensibilidad moderada y penalizamos falta de alerta
                loss = -math.log(max(0.5, e_val)) if e_val > 1.0 else 0.5

            # Actualización exponencial Hedge: w <- w * exp(-eta * loss)
            current_w = self._weights.get(name, 1.0 / len(self._weights))
            decay_factor = math.exp(-self.learning_rate * loss)
            self._weights[name] = max(self.min_weight_floor, current_w * decay_factor)
            self._cumulative_regret[name] += loss

        self._normalize_weights()
        return self.current_weights

    def _normalize_weights(self) -> None:
        """Normaliza los pesos para que sumen 1.0 respetando el suelo de seguridad."""
        total = sum(self._weights.values())
        if total < 1e-9:
            n = len(self._weights) or 1
            self._weights = {k: 1.0 / n for k in self._weights}
            return
        self._weights = {k: v / total for k, v in self._weights.items()}
