"""Observatory learning curve and evidence requirement."""

from __future__ import annotations

from collections.abc import Sequence

from ..adaptation.guard import wilson_lower_bound
from .types import ObservationRow, LearningPoint, ContextLearning
from .helpers import evaluated, chronological


def learning_curve(
    rows: Sequence[ObservationRow],
    *,
    targets: Sequence[int] = (20, 100, 500, 1000, 5000, 10000),
    z: float = 1.96,
) -> tuple[LearningPoint, ...]:
    """Accuracy acumulada (por orden cronológico) en cada objetivo de n.

    Si el contexto tiene menos observaciones que un objetivo, el último
    punto reporta el total disponible (el dashboard lo marca como
    insuficiente).
    """
    evaluated_rows = chronological(evaluated(rows))
    total = len(evaluated_rows)
    if total == 0:
        return ()
    points: list[LearningPoint] = []
    hits = 0
    prev = 0
    for target in targets:
        n = min(target, total)
        if n <= prev:
            continue
        hits += sum(
            1 for r in evaluated_rows[prev:n] if r.direction_correct
        )
        prev = n
        points.append(
            LearningPoint(
                n=n,
                accuracy=hits / n,
                wilson_lb=wilson_lower_bound(hits, n, z=z),
            )
        )
        if n == total:
            break
    return tuple(points)


def evidence_requirement(
    rows: Sequence[ObservationRow],
    *,
    min_accuracy: float = 0.52,
    step: int = 20,
    z: float = 1.96,
    max_n: int | None = None,
) -> int | None:
    """n mínimo donde el Wilson 95% de la accuracy acumulada >= umbral.

    Es la respuesta a "¿cuánto necesita aprender ZENIN antes de poder
    confiar en sí mismo en este contexto?". Cruce ESTABLE: el n más
    pequeño (grilla de ``step`` o el punto final parcial) donde el
    Wilson 95% supera el umbral Y se mantiene por encima en todos los
    puntos posteriores — el edge de una muestra pequeña que colapsa
    después NO cuenta como evidencia (lección de 9.5). Devuelve None
    si nunca lo alcanza de forma estable.
    """
    if not 0.0 < min_accuracy < 1.0:
        raise ValueError(f"min_accuracy inválida: {min_accuracy}")
    if step < 1:
        raise ValueError(f"step inválido: {step}")
    evaluated_rows = chronological(evaluated(rows))
    total = len(evaluated_rows)
    limit = min(total, max_n) if max_n is not None else total
    if limit < 1:
        return None
    grid: list[int] = list(range(step, limit + 1, step))
    if not grid or grid[-1] != limit:
        grid.append(limit)
    bounds: list[tuple[int, float]] = []
    hits = 0
    for n in grid:
        start = bounds[-1][0] if bounds else 0
        hits += sum(
            1 for r in evaluated_rows[start:n] if r.direction_correct
        )
        bounds.append((n, wilson_lower_bound(hits, n, z=z)))
    for i, (n, lb) in enumerate(bounds):
        if lb < min_accuracy:
            continue
        if all(later >= min_accuracy for _, later in bounds[i:]):
            return n
    return None