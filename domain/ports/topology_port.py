"""Puertos de contrato para topología causal, estimación online y orquestación de supresión de cascadas.

Puro: solo biblioteca estándar (typing.Protocol, typing.Mapping, typing.Sequence).
Sin dependencias de infraestructura ni librerías de terceros (numpy, scipy).
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from ..entities.consensus import ConsensusDecision
from ..entities.topology import CausalEdge, SystemWideAlarm

# Alias de retrocompatibilidad
AdaptiveGateDecision = ConsensusDecision


@runtime_checkable
class CausalTopologyPort(Protocol):
    """Contrato puro para la estimación y consulta del grafo causal dinámico."""

    def register_pair_observation(
        self,
        source_id: str,
        target_id: str,
        source_value: float,
        target_value: float,
        step: int,
    ) -> None:
        """Actualiza la estimación online de entropía de transferencia / correlación con retardo."""
        ...

    def get_active_edges(self) -> Sequence[CausalEdge]:
        """Retorna las aristas causales activas que superan el umbral de acoplamiento."""
        ...


@runtime_checkable
class CausalGatingOrchestratorPort(Protocol):
    """Contrato del orquestador que suprime alertas en cascada y genera SystemWideAlarm."""

    def process_decisions(
        self,
        step: int,
        node_decisions: Mapping[str, ConsensusDecision],
    ) -> Sequence[SystemWideAlarm]:
        """Unifica alertas individuales, suprime efectos retardados y emite el RCA consolidado."""
        ...
