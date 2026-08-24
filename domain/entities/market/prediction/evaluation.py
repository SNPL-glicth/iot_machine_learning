"""Evaluación de una predicción contra su Outcome (FASE 3).

Funciones puras: sin estado, sin I/O. La evaluación mide dirección,
magnitud, cobertura del intervalo y calibración de la probabilidad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .outcome import Outcome
    from .prediction import Prediction, PredictionInterval

_EPS = 1e-12


@dataclass(frozen=True, slots=True, kw_only=True)
class Evaluation:
    """Resultado de comparar una predicción con su desenlace.

    Attributes:
        direction_correct: ``True`` si el retorno realizado tuvo el signo
            esperado (expected_return >= 0 -> subida; < 0 -> caída).
        magnitude_error: |retorno realizado - retorno esperado|.
        within_interval: ``True`` si el retorno realizado cayó en el
            intervalo de la predicción (``False`` sin intervalo).
        calibration_error: |probability_up - acierto| (0..1), donde
            acierto = 1 si el retorno fue positivo, 0 si no.
    """

    direction_correct: bool
    magnitude_error: float
    within_interval: bool
    calibration_error: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.magnitude_error) or self.magnitude_error < 0:
            raise ValueError(f"magnitude_error inválido: {self.magnitude_error!r}")
        if not math.isfinite(self.calibration_error) or not (
            0.0 <= self.calibration_error <= 1.0
        ):
            raise ValueError(
                f"calibration_error fuera de [0, 1]: {self.calibration_error!r}"
            )


def direction_correct(expected_return: float, return_realized: float, probability_up: float = 0.5) -> bool:
    """Acierto direccional: signo esperado contra signo realizado.

    Convenio: expected_return != 0 usa su signo; si expected_return == 0,
    usa probability_up >= 0.5 para determinar la dirección esperada.
    """
    if abs(expected_return) > 1e-12:
        # Usar signo de expected_return si es significativamente no-cero
        if expected_return >= 0.0:
            return return_realized >= 0.0
        return return_realized < 0.0
    # Fallback: usar probability_up >= 0.5
    if probability_up >= 0.5:
        return return_realized >= 0.0
    return return_realized < 0.0


def magnitude_error(expected_return: float, return_realized: float) -> float:
    """Error absoluto de magnitud."""
    return abs(return_realized - expected_return)


def within_interval(
    interval: PredictionInterval | None, return_realized: float
) -> bool:
    """Check de cobertura para mypy: el caller pasa el intervalo real."""
    """``True`` si el retorno realizado cae dentro del intervalo."""
    if interval is None:
        return False
    lower = interval.lower
    upper = interval.upper
    return lower <= return_realized <= upper


def calibration_error(probability_up: float, return_realized: float) -> float:
    """Desviación de la probabilidad contra el acierto observado (0..1)."""
    hit = 1.0 if return_realized > _EPS else 0.0
    return abs(probability_up - hit)


def evaluate_prediction(prediction: Prediction, outcome: Outcome) -> Evaluation:
    """Evalúa una predicción contra su Outcome.

    Validaciones de asociación (mismo símbolo y mismo horizonte) se
    realizan en ``Prediction._guard_outcome`` antes de llamar aquí; esta
    función es la matemática pura y asume predicción/outcome alineados.
    """
    return Evaluation(
        direction_correct=direction_correct(
            prediction.expected_return, outcome.return_realized, prediction.probability_up
        ),
        magnitude_error=magnitude_error(
            prediction.expected_return, outcome.return_realized
        ),
        within_interval=within_interval(prediction.interval, outcome.return_realized),
        calibration_error=calibration_error(
            prediction.probability_up, outcome.return_realized
        ),
    )
