"""Entidades y tipos inmutables para la capa de representación y evidencia asimétrica.

Puro: solo biblioteca estándar (dataclasses, enum, typing).
Sin dependencias de infraestructura ni librerías de terceros (numpy/scipy).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class RepresentationLevel(Enum):
    """Niveles de resolución / representación invariante (Klein)."""

    TEN_X = "10X"  # Envolvente / extrema compresión / reposo
    TWO_X = "2X"  # Nivel / cambio estructural / deriva
    RAW = "RAW"  # Resolución completa / choque / alta frecuencia


class SystemOperationalState(Enum):
    """Estado operativo inferido por la política."""

    RESTING = auto()  # Operación nominal sin perturbaciones
    DRIFTING = auto()  # Sospecha o confirmación de cambio de régimen persistente
    SHOCKED = auto()  # Disrupción instantánea / choque abrupto


@dataclass(frozen=True)
class PolicyDecision:
    """Decisión inmutable emitida por el enrutador de representación."""

    level: RepresentationLevel
    operational_state: SystemOperationalState
    reason: str
    backfill_count: int = 0
    flapping_suppressed: bool = False


@dataclass(frozen=True)
class EvidenceScore:
    """Unidad atómica de evidencia emitida por un experto asimétrico."""

    expert_name: str
    representation_affinity: RepresentationLevel
    anomaly_probability: float  # [0.0, 1.0] o log-likelihood ratio
    compute_cost_estimate: float  # Tiempo/flops normalizados
    metadata: dict[str, Any] | None = None
