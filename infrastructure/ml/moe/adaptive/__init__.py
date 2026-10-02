"""Módulo de Adaptive Meta-Gating por Consenso de Kuramoto (Fase 2.5).

Proporciona compuertas de decisión no lineales por sincronización topológica de fase
con integración Euler de campo medio O(N) y quenching refractario anti-histéresis.
"""

from __future__ import annotations

from .kuramoto_gate import KuramotoConsensusGate

__all__ = [
    "KuramotoConsensusGate",
]
