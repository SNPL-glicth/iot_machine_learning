"""Calibración de probabilidades (FASE 7.5) — ¿P(up) declarado ≈ acierto real?

La pregunta que responde el dashboard y, más adelante, la primera etapa
de aprendizaje (FASE 8): si ZENIN dice P=0.9, ¿históricamente acertó
~90% o ~55%?

Toda la lógica es pura (dominio, sin SQL): recibe por bucket la
probabilidad declarada promedio, aciertos y tamaño de muestra, y
produce:

- estado del bucket: OK / INSUFFICIENT (n < umbral: no concluye) / FAIL
  (desviación > tolerancia: calibración rota);
- ECE (Expected Calibration Error) ponderado por muestra;
- curva de calibración ASCII (declarado vs realizado vs diagonal).

Los umbrales (``min_n``, ``tolerance``) son el germen de los guardrails
de FASE 8: no se concluye con muestras diminutas, no se aprende de un
único resultado.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "BucketStatus",
    "BucketReport",
    "CalibrationThresholds",
    "CalibrationReport",
    "bucket_calibration",
    "calibration_chart",
]


class BucketStatus(Enum):
    """Estado de calibración de un bucket de confianza."""

    OK = "ok"
    INSUFFICIENT = "insufficient"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class CalibrationThresholds:
    """Guardrails de conclusión para la curva de calibración."""

    min_n: int = 5
    tolerance: float = 0.10

    def __post_init__(self) -> None:
        if self.min_n < 1:
            raise ValueError(f"min_n debe ser >= 1: {self.min_n}")
        if not 0.0 < self.tolerance <= 1.0:
            raise ValueError(f"tolerance inválida: {self.tolerance}")


@dataclass(frozen=True, slots=True)
class BucketReport:
    """Resultado por bucket de confianza (inmutable)."""

    label: str
    declared: float  # probabilidad declarada promedio (P(up))
    n: int  # predicciones evaluadas en el bucket
    hits: int
    hit_rate: float  # acierto real observado (hits / n)
    delta: float  # declared - hit_rate
    status: BucketStatus

    @property
    def ece_contribution(self) -> float:
        return abs(self.delta) * self.n


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Reporte completo de la curva de calibración."""

    buckets: tuple[BucketReport, ...]
    thresholds: CalibrationThresholds

    @property
    def ece(self) -> float:
        """Expected Calibration Error, ponderado por el tamaño de muestra."""
        total = sum(b.n for b in self.buckets)
        if total == 0:
            return 0.0
        return sum(b.ece_contribution for b in self.buckets) / total

    @property
    def failing_buckets(self) -> tuple[BucketReport, ...]:
        return tuple(b for b in self.buckets if b.status is BucketStatus.FAIL)

    @property
    def insufficient_buckets(self) -> tuple[BucketReport, ...]:
        return tuple(b for b in self.buckets if b.status is BucketStatus.INSUFFICIENT)

    @property
    def has_failures(self) -> bool:
        return bool(self.failing_buckets)


def bucket_calibration(
    samples: Iterable[tuple[str, float, int, int]],
    thresholds: CalibrationThresholds | None = None,
) -> CalibrationReport:
    """Clasifica buckets de confianza por su calibración.

    Args:
        samples: iterable de ``(label, declared, hits, n)`` — la
            probabilidad declarada promedio, los aciertos y el tamaño de
            muestra de cada bucket (ordenado de menor a mayor confianza).
        thresholds: umbrales ``min_n``/``tolerance`` (ver clase).
    """
    thresholds = thresholds or CalibrationThresholds()
    buckets: list[BucketReport] = []
    for label, declared, hits, n in samples:
        if n < 0 or hits < 0 or hits > n:
            raise ValueError(
                f"muestra inválida para {label!r}: declared={declared} "
                f"hits={hits} n={n}"
            )
        if not 0.0 <= declared <= 1.0:
            raise ValueError(f"declared fuera de [0,1] para {label!r}: {declared}")
        hit_rate = hits / n if n else 0.0
        delta = declared - hit_rate
        if n < thresholds.min_n:
            status = BucketStatus.INSUFFICIENT
        elif abs(delta) <= thresholds.tolerance:
            status = BucketStatus.OK
        else:
            status = BucketStatus.FAIL
        buckets.append(
            BucketReport(
                label=label,
                declared=declared,
                n=n,
                hits=hits,
                hit_rate=hit_rate,
                delta=delta,
                status=status,
            )
        )
    return CalibrationReport(
        buckets=tuple(buckets),
        thresholds=thresholds,
    )


def calibration_chart(
    report: CalibrationReport,
    *,
    width: int = 42,
    height: int = 11,
) -> str:
    """Curva de calibración ASCII.

    Diagonal (``\\``) = calibración perfecta (declarado == realizado).
    ``o`` = declarado, ``x`` = realizado (hits/n), ``*`` = coinciden.
    Solo se grafican buckets con n >= 1.
    """
    if width < 10 or height < 5:
        raise ValueError("curva demasiado pequeña: width>=10, height>=5")

    points = [
        b for b in report.buckets if b.n >= 1 and 0.0 <= b.declared <= 1.0
    ]
    if not points:
        return "(sin datos suficientes para la curva)"

    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0

    def _col(value: float) -> int:
        return round((value - x_min) / (x_max - x_min) * (width - 1))

    def _row(value: float) -> int:
        return round((y_max - value) / (y_max - y_min) * (height - 1))

    grid: list[list[set[str]]] = [
        [set() for _ in range(width)] for _ in range(height)
    ]
    for b in points:
        grid[_row(b.declared)][_col(b.declared)].add("o")
        grid[_row(b.hit_rate)][_col(b.hit_rate)].add("x")

    for i in range(height):
        diag_col = round(i / (height - 1) * (width - 1)) if height > 1 else 0
        grid[i][diag_col].add("\\")

    lines: list[str] = []
    for row in range(height):
        y = y_max - row * (y_max - y_min) / (height - 1)
        cells: list[str] = []
        for col in range(width):
            markers = grid[row][col]
            if not markers:
                cells.append(" ")
            elif len(markers) == 1:
                cells.append(markers.pop())
            elif len(markers) == 2 and "\\" in markers and {"o", "x"} == (markers - {"\\"}):
                cells.append("*")
            else:
                cells.append("*")
        lines.append(f"{y:4.1f} |{''.join(cells)}|")

    axis = "    +" + "-" * width + "+"
    lines.append(axis)
    x_labels = (
        f"{x_min:4.1f} "
        + " " * (width // 2 - 4)
        + f"{0.5:4.1f}"
        + " " * (width // 2 - 4 + (width % 2))
        + f"{x_max:4.1f}"
    )
    lines.append(x_labels)
    lines.append("     declarado P(up) ->  (o=declarado, x=realizado, "
                 "\\=diagonal ideal)")
    return "\n".join(lines)
