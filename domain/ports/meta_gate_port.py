"""Puertos de contrato para calibración online y compuerta de decisión adaptativa (Meta-Gating).

Puro: solo biblioteca estándar (typing.Protocol, typing.Mapping, typing.Sequence).
Sin dependencias de infraestructura ni librerías de terceros (numpy, scipy).
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from ..entities.conformal_risk import AdaptiveGateDecision, ConformalBound
from ..entities.representation_evidence import EvidenceScore, SystemOperationalState


@runtime_checkable
class OnlineCalibratorPort(Protocol):
    """Contrato puro para la ponderación secuencial adaptativa de expertos (Hedge / OCO)."""

    def compute_e_values(
        self, evidences: Sequence[EvidenceScore]
    ) -> Sequence[float]:
        """Transforma puntuaciones atómicas o log-likelihood ratios en e-values (E[e|H0] <= 1)."""
        ...

    def update_weights(
        self,
        evidences: Sequence[EvidenceScore],
        e_values: Sequence[float],
        current_state: SystemOperationalState,
    ) -> Mapping[str, float]:
        """Actualiza los pesos convexos w_{i, t+1} según la pérdida/regret observado."""
        ...

    @property
    def current_weights(self) -> Mapping[str, float]:
        """Retorna el mapeo inmutable de pesos activos de los expertos."""
        ...


@runtime_checkable
class AdaptiveMetaGatePort(Protocol):
    """Contrato para la integración de evidencias y evaluación del stopping time certificado."""

    def evaluate_step(
        self,
        step: int,
        evidences: Sequence[EvidenceScore],
        operational_state: SystemOperationalState,
        budget_remaining_ratio: float = 1.0,
    ) -> AdaptiveGateDecision:
        """Calcula el e-value combinado, actualiza la martingala M_t y resuelve el disparo adaptativo."""
        ...

    def reset_martingale(self) -> None:
        """Reinicia el proceso martingala a M_0 = 1.0 tras una alarma certificada o reinicio de régimen."""
        ...

    @property
    def bounds(self) -> ConformalBound:
        """Retorna las cotas de riesgo activas del sistema."""
        ...
