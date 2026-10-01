"""Puerto de contrato para expertos del MoE asimétrico.

Puro: solo biblioteca estándar (typing.Protocol, typing.Sequence).
Declara la afinidad de escala (10X, 2X, RAW) para el enrutamiento selectivo.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..entities.representation_evidence import EvidenceScore, RepresentationLevel


@runtime_checkable
class AsymmetricExpertPort(Protocol):
    """Contrato que debe cumplir todo experto del MoE asimétrico."""

    name: str
    affinity: RepresentationLevel

    def evaluate(self, series_slice: Sequence[float]) -> EvidenceScore:
        """Calcula la evidencia de anomalía dado un slice en su escala nativa."""
        ...
