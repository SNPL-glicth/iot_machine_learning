"""Traza de razonamiento estructurada del motor cognitivo.

Documenta cada fase del pipeline cognitivo como un ``ReasoningPhase``
y las agrupa en un ``ReasoningTrace`` ordenado cronológicamente.

Pipeline cognitivo:
    1. perceive  — análisis de señal
    2. filter    — filtrado aplicado
    3. predict   — predicciones individuales
    4. inhibit   — supresión de engines inestables
    5. adapt     — ajuste de pesos por plasticidad
    6. fuse      — fusión ponderada

Cada fase registra: qué se hizo, qué entró, qué salió, y cuánto
tardó (opcional).

Domain-pure.  Sin dependencias de infraestructura.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PhaseKind(str, Enum):
    """Fases del pipeline cognitivo."""

    PERCEIVE = "perceive"
    FILTER = "filter"
    PREDICT = "predict"
    INHIBIT = "inhibit"
    ADAPT = "adapt"
    FUSE = "fuse"


@dataclass(frozen=True)
class ReasoningPhase:
    """Una fase individual del razonamiento cognitivo.

    Attributes:
        kind: Tipo de fase (``PhaseKind``).
        summary: Resumen estructurado (clave-valor, no texto humano).
        inputs: Datos de entrada a la fase.
        outputs: Datos de salida de la fase.
        duration_ms: Duración en milisegundos (opcional).
        metadata: Datos adicionales específicos de la fase.
    """

    kind: PhaseKind
    summary: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "kind": self.kind.value,
            "summary": self.summary,
            "inputs": _safe_serialize(self.inputs),
            "outputs": _safe_serialize(self.outputs),
        }
        if self.duration_ms is not None:
            d["duration_ms"] = round(self.duration_ms, 3)
        if self.metadata:
            d["metadata"] = _safe_serialize(self.metadata)
        return d


@dataclass(frozen=True)
class ReasoningTrace:
    """Traza completa de razonamiento del motor cognitivo.

    Secuencia ordenada de fases que documenta el proceso de
    decisión de principio a fin.

    Attributes:
        phases: Fases en orden cronológico.
        total_duration_ms: Duración total del pipeline (opcional).
        regime_at_inference: Régimen detectado al momento de inferir.
        n_engines_available: Engines disponibles al inicio.
        n_engines_active: Engines que participaron (no inhibidos).
    """

    phases: List[ReasoningPhase] = field(default_factory=list)
    total_duration_ms: Optional[float] = None
    regime_at_inference: str = "unknown"
    n_engines_available: int = 0
    n_engines_active: int = 0

    @property
    def phase_kinds(self) -> List[str]:
        """Nombres de las fases ejecutadas, en orden."""
        return [p.kind.value for p in self.phases]

    @property
    def has_inhibition(self) -> bool:
        """True si alguna fase de inhibición fue ejecutada."""
        return any(p.kind == PhaseKind.INHIBIT for p in self.phases)

    @property
    def has_adaptation(self) -> bool:
        """True si la fase de adaptación (plasticidad) fue ejecutada."""
        return any(p.kind == PhaseKind.ADAPT for p in self.phases)

    def get_phase(self, kind: PhaseKind) -> Optional[ReasoningPhase]:
        """Obtiene la primera fase del tipo dado."""
        for p in self.phases:
            if p.kind == kind:
                return p
        return None

    def to_dict(self) -> dict:
        d: dict = {
            "phases": [p.to_dict() for p in self.phases],
            "phase_kinds": self.phase_kinds,
            "regime_at_inference": self.regime_at_inference,
            "n_engines_available": self.n_engines_available,
            "n_engines_active": self.n_engines_active,
        }
        if self.total_duration_ms is not None:
            d["total_duration_ms"] = round(self.total_duration_ms, 3)
        return d

    @classmethod
    def empty(cls) -> ReasoningTrace:
        return cls()


def _safe_serialize(data: Dict[str, Any]) -> dict:
    """Serializa un dict asegurando que los valores son JSON-safe.

    Convierte objetos con ``to_dict()`` y deja pasar primitivos.
    """
    result: dict = {}
    for k, v in data.items():
        if hasattr(v, "to_dict"):
            result[k] = v.to_dict()
        elif isinstance(v, (list, tuple)):
            result[k] = [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in v
            ]
        elif isinstance(v, dict):
            result[k] = _safe_serialize(v)
        else:
            result[k] = v
    return result
