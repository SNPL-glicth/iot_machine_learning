"""Taxonomía del error de predicción (FASE 7).

No "prediction → trade → profit/loss" sino: ¿QUÉ información estaba
equivocada? El error se atribuye a causas ordenadas por prioridad
documentada; la primera que dispara es la primaria, todas las que
disparan quedan como contribuyentes (auditoría completa, sin
diagnóstico único inventado).

Prioridad (la evidencia más accionable primero):
    1. TAIL_EVENT — el realizado perforó la cola declarada (cvar).
    2. REGIME_SHIFT — cambió el régimen y falló la dirección.
    3. DATA_DEGRADED — dato degradado + error material.
    4. COST_KILL — dirección correcta pero el neto no paga.
    5. OVERCONFIDENCE — falló con confianza alta.
    6. CALIBRATION_DRIFT — acertó la dirección pero la P estaba rota.
    7. SIGNAL_DECAY — falló la dirección (causa por defecto).
    8. QUIET — acertó y ajustado: nada que aprender.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ErrorCause",
    "ErrorAttribution",
    "attribute_error",
    "OVERCONFIDENCE_THRESHOLD",
    "CALIBRATION_DRIFT_THRESHOLD",
    "DEGRADED_MAGNITUDE_TOL",
]

OVERCONFIDENCE_THRESHOLD: float = 0.8
CALIBRATION_DRIFT_THRESHOLD: float = 0.5
DEGRADED_MAGNITUDE_TOL: float = 0.005


class ErrorCause(str, Enum):
    """Causas atribuibles (valores estables: persisten en el ledger)."""

    TAIL_EVENT = "tail_event"
    REGIME_SHIFT = "regime_shift"
    DATA_DEGRADED = "data_degraded"
    COST_KILL = "cost_kill"
    OVERCONFIDENCE = "overconfidence"
    CALIBRATION_DRIFT = "calibration_drift"
    SIGNAL_DECAY = "signal_decay"
    QUIET = "quiet"


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorAttribution:
    """Atribución de un error (inmutable, con rastro)."""

    primary: ErrorCause
    contributors: tuple[ErrorCause, ...]
    magnitude_error: float
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.primary, ErrorCause):
            raise TypeError("primary debe ser ErrorCause")
        if self.primary not in self.contributors:
            raise ValueError("primary debe estar en contributors")
        if not math.isfinite(self.magnitude_error) or self.magnitude_error < 0:
            raise ValueError("magnitude_error inválido")


def attribute_error(
    *,
    direction_correct: bool,
    magnitude_error: float,
    calibration_error: float,
    confidence: float,
    net_return: float,
    tail_breach: bool = False,
    regime_at_predict: str | None = None,
    regime_at_resolve: str | None = None,
    data_degraded: bool = False,
) -> ErrorAttribution:
    """Atribuye el error a causas (puro, determinista)."""
    if not math.isfinite(magnitude_error) or magnitude_error < 0:
        raise ValueError("magnitude_error inválido")
    if not math.isfinite(calibration_error) or not (
        0.0 <= calibration_error <= 1.0
    ):
        raise ValueError("calibration_error fuera de [0, 1]")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence fuera de [0, 1]")
    if not math.isfinite(net_return):
        raise ValueError("net_return no finito")

    regime_changed = (
        regime_at_predict is not None
        and regime_at_resolve is not None
        and regime_at_predict != regime_at_resolve
    )
    degraded_material = data_degraded and (
        not direction_correct or magnitude_error > DEGRADED_MAGNITUDE_TOL
    )

    triggered: list[ErrorCause] = []
    if tail_breach:
        triggered.append(ErrorCause.TAIL_EVENT)
    if regime_changed and not direction_correct:
        triggered.append(ErrorCause.REGIME_SHIFT)
    if degraded_material:
        triggered.append(ErrorCause.DATA_DEGRADED)
    if direction_correct and net_return <= 0:
        triggered.append(ErrorCause.COST_KILL)
    if not direction_correct and confidence >= OVERCONFIDENCE_THRESHOLD:
        triggered.append(ErrorCause.OVERCONFIDENCE)
    if direction_correct and calibration_error >= CALIBRATION_DRIFT_THRESHOLD:
        triggered.append(ErrorCause.CALIBRATION_DRIFT)
    if not direction_correct:
        triggered.append(ErrorCause.SIGNAL_DECAY)
    if not triggered:
        triggered.append(ErrorCause.QUIET)

    primary = triggered[0]
    detail = (
        f"{primary.value}: mag_err={magnitude_error:.4f} "
        f"net={net_return:+.4f} "
        f"régimen {regime_at_predict or '?'}→{regime_at_resolve or '?'}"
    )
    return ErrorAttribution(
        primary=primary,
        contributors=tuple(triggered),
        magnitude_error=magnitude_error,
        detail=detail,
    )
