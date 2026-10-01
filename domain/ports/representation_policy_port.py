"""Puertos de contrato para la política de representación y centinelas.

Puro: solo biblioteca estándar (typing.Protocol, typing.Sequence).
Sin dependencias de infraestructura ni librerías externas.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..entities.representation_evidence import PolicyDecision, RepresentationLevel


@runtime_checkable
class BaseSentinelPort(Protocol):
    """Contrato para centinelas de detección rápida y evaluación de sorpresas."""

    def inspect(self, current_value: float, delta: float) -> bool:
        """Determina si se violó el contrato empírico o cuantil nominal."""
        ...


@runtime_checkable
class BaseRepresentationPolicyPort(Protocol):
    """Contrato del enrutador de representación adaptativa del stream."""

    def step(self, point: float, index: int) -> PolicyDecision:
        """Evalúa un nuevo punto del stream y decide la representación adecuada."""
        ...

    def get_effective_stream_slice(self) -> tuple[RepresentationLevel, Sequence[float]]:
        """Retorna el slice temporal adaptado a la representación activa."""
        ...
