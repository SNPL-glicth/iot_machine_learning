"""Módulo de Mixture of Experts Asimétrico (Fase 1).

Enrutamiento por afinidad de representación temporal, desacoplamiento estricto
del ciclo de inferencia y acumulación secuencial de evidencia (Anytime Martingale).
"""

from __future__ import annotations

from .accumulator import EvidenceAccumulator, IntegratedEvidence
from .catalog import (
    HighFrequencyExpert,
    RegimeShiftExpert,
    RestingInvariantExpert,
)
from .dispatcher import AsymmetricDispatcher

__all__ = [
    "AsymmetricDispatcher",
    "EvidenceAccumulator",
    "IntegratedEvidence",
    "RestingInvariantExpert",
    "RegimeShiftExpert",
    "HighFrequencyExpert",
]
