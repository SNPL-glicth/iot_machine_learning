"""Puertos de contrato para compuerta de decisión por sincronización de fase (Consensus Gate).

Puro: solo biblioteca estándar (typing.Protocol, typing.Sequence, typing.runtime_checkable).
Sin dependencias de infraestructura ni librerías de terceros (numpy, scipy).
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..entities.consensus import ConsensusDecision
from ..entities.representation_evidence import EvidenceScore, SystemOperationalState


@runtime_checkable
class ConsensusGatePort(Protocol):
    """Contrato puro para la compuerta de decisión por sincronización topológica de fase."""

    def evaluate_step(
        self,
        step: int,
        evidences: Sequence[EvidenceScore],
        operational_state: SystemOperationalState,
        budget_remaining_ratio: float = 1.0,
    ) -> ConsensusDecision:
        """Integra las evidencias angulares y evalúa la transición de fase en el paso step."""
        ...

    def reset(self) -> None:
        """Reinicia el estado de fases y cuenta regresiva de quenching."""
        ...


# Alias de retrocompatibilidad arquitectónica
AdaptiveMetaGatePort = ConsensusGatePort
