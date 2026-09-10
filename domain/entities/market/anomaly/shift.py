"""Veredicto de distribution shift por voto ponderado (FASE 4).

Filosofía del ensemble Rosa Roja: 7 detectores votan con peso, nadie
decide solo. Aquí 6 señales (4 continuas + changepoint + staleness
binarias) votan con pesos documentados; SHIFT si el peso en breach
alcanza el quórum.

Pesos: volumen 0.25, spread 0.20, volatilidad 0.25, flujo 0.15,
changepoint 0.10, staleness 0.05. Quórum default 0.40: dos señales
fuertes cualesquiera (o volumen+volatilidad) vetan; una sola no.
"""

from __future__ import annotations

from dataclasses import dataclass

from .scores import AnomalyScore

__all__ = [
    "SHIFT_WEIGHTS",
    "SHIFT_QUORUM",
    "ShiftVerdict",
    "detect_shift",
    "binary_score",
]

#: Pesos por señal (suman 1.0).
SHIFT_WEIGHTS: dict[str, float] = {
    "volume_spike": 0.25,
    "spread_shock": 0.20,
    "vol_explosion": 0.25,
    "flow_extreme": 0.15,
    "changepoint": 0.10,
    "staleness": 0.05,
}

#: Peso en breach necesario para declarar SHIFT.
SHIFT_QUORUM: float = 0.40


@dataclass(frozen=True, slots=True, kw_only=True)
class ShiftVerdict:
    """Veredicto de shift (inmutable, con contribuyentes para auditoría)."""

    is_shift: bool
    weight: float  # peso total en breach [0, 1]
    quorum: float
    contributors: tuple[str, ...]  # nombres en breach, ordenados por peso
    n_breached: int
    n_total: int

    @property
    def summary(self) -> str:
        state = "SHIFT" if self.is_shift else "normal"
        who = ",".join(self.contributors) if self.contributors else "—"
        return f"{state} (peso {self.weight:.2f}/{self.quorum:.2f}: {who})"


def binary_score(name: str, triggered: bool) -> AnomalyScore:
    """Envuelve una señal binaria (changepoint, staleness) como score."""
    if name not in SHIFT_WEIGHTS:
        raise ValueError(f"señal binaria desconocida: {name!r}")
    return AnomalyScore(
        name=name, score=1.0 if triggered else 0.0, threshold=0.5,
        breached=triggered,
        detail="disparada" if triggered else "en calma",
    )


def detect_shift(
    scores: tuple[AnomalyScore, ...] | list[AnomalyScore],
    *,
    quorum: float = SHIFT_QUORUM,
) -> ShiftVerdict:
    """Voto ponderado sobre los scores (nombres duplicados fallan)."""
    scores = tuple(scores)
    if not scores:
        raise ValueError("sin scores no hay veredicto")
    names = [s.name for s in scores]
    if len(set(names)) != len(names):
        raise ValueError(f"scores duplicados: {names!r}")
    unknown = [n for n in names if n not in SHIFT_WEIGHTS]
    if unknown:
        raise ValueError(f"señales desconocidas: {unknown!r}")
    if not 0.0 < quorum <= 1.0:
        raise ValueError(f"quorum fuera de (0, 1]: {quorum!r}")

    breached = [s for s in scores if s.breached]
    weight = sum(SHIFT_WEIGHTS[s.name] for s in breached)
    ordered = tuple(
        s.name for s in sorted(
            breached, key=lambda s: SHIFT_WEIGHTS[s.name], reverse=True
        )
    )
    return ShiftVerdict(
        is_shift=weight >= quorum,
        weight=round(weight, 4),
        quorum=quorum,
        contributors=ordered,
        n_breached=len(breached),
        n_total=len(scores),
    )
