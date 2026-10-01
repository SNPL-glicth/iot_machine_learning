"""Entidades inmutables para topología multivariada, grafos causales y atribución de causa raíz (RCA).

Puro: solo biblioteca estándar (dataclasses, enum, typing).
Sin dependencias de infraestructura ni librerías de terceros (numpy, scipy, torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Mapping

from .representation_evidence import SystemOperationalState


class CausalRelationType(Enum):
    """Naturaleza de la conexión dirigida entre series temporales."""

    DIRECT_LEAD = auto()     # X precede directamente a Y con retardo delta
    COMMON_CAUSE = auto()    # X e Y son excitados por un nodo común/oculto
    SPURIOUS_NOISE = auto()  # Co-ocurrencia no informativa / correlación espuria


@dataclass(frozen=True)
class CausalEdge:
    """Arista causal dirigida e inmutable en el grafo topológico."""

    source_series_id: str
    target_series_id: str
    lag_steps: int                    # Retardo temporal característico delta >= 1
    coupling_strength: float          # Transfer entropy o score normalizado [0.0, 1.0]
    relation_type: CausalRelationType = CausalRelationType.DIRECT_LEAD


@dataclass(frozen=True)
class RootCauseDiagnosis:
    """Diagnóstico simbólico inmutable del nodo y mecanismo origen de la perturbación."""

    root_series_id: str
    dominant_mechanism: str           # e.g. "regime_shift_2x", "high_frequency_raw"
    mechanism_confidence: float       # Peso convexo del experto dominante w_{i, t} in [0, 1]
    detection_step: int
    operational_state: SystemOperationalState
    active_expert_weights: Mapping[str, float]


@dataclass(frozen=True)
class SystemWideAlarm:
    """Alerta unificada de subsistema que consolida la cascada de fallas multivariada."""

    alarm_id: str
    root_cause: RootCauseDiagnosis
    affected_series_ids: tuple[str, ...]
    suppressed_cascade_count: int
    peak_martingale_value: float
    total_compute_saved_by_suppression: float
    explanation_summary: str
