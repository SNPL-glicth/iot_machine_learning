"""Directivas de adaptación desde la atribución (FASE 7).

Propuesta ≠ update (patrón FASE 8): la directiva DICE qué haría y por
qué, con los números reales del error. Nada se aplica en vivo — el
refit offline acepta/rechaza y versiona (append-only).

Mapa causa → acción:
    TAIL_EVENT → WIDEN_TAILS (colas declaradas demasiado finas)
    REGIME_SHIFT → REFIT_REGIME (transición no aprendida)
    DATA_DEGRADED → QUARANTINE_FEED (proveedor/tape en cuarentena)
    COST_KILL → REVIEW_COSTS (el edge no paga: subir costos o callar)
    OVERCONFIDENCE → RECALIBRATE (confianza rota)
    CALIBRATION_DRIFT → RECALIBRATE (P rota aunque el signo acertó)
    SIGNAL_DECAY → DOWNWEIGHT_EXPERT (la señal dejó de funcionar)
    QUIET → NO_ACTION
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .error_taxonomy import ErrorAttribution, ErrorCause

__all__ = ["AdaptationAction", "AdaptationDirective", "propose_directive"]


class AdaptationAction(str, Enum):
    """Acciones proponibles (valores estables: persisten en el ledger)."""

    WIDEN_TAILS = "widen_tails"
    REFIT_REGIME = "refit_regime"
    QUARANTINE_FEED = "quarantine_feed"
    REVIEW_COSTS = "review_costs"
    RECALIBRATE = "recalibrate"
    DOWNWEIGHT_EXPERT = "downweight_expert"
    NO_ACTION = "no_action"


_CAUSE_TO_ACTION: dict[ErrorCause, AdaptationAction] = {
    ErrorCause.TAIL_EVENT: AdaptationAction.WIDEN_TAILS,
    ErrorCause.REGIME_SHIFT: AdaptationAction.REFIT_REGIME,
    ErrorCause.DATA_DEGRADED: AdaptationAction.QUARANTINE_FEED,
    ErrorCause.COST_KILL: AdaptationAction.REVIEW_COSTS,
    ErrorCause.OVERCONFIDENCE: AdaptationAction.RECALIBRATE,
    ErrorCause.CALIBRATION_DRIFT: AdaptationAction.RECALIBRATE,
    ErrorCause.SIGNAL_DECAY: AdaptationAction.DOWNWEIGHT_EXPERT,
    ErrorCause.QUIET: AdaptationAction.NO_ACTION,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class AdaptationDirective:
    """Propuesta de adaptación (inmutable, auditable)."""

    action: AdaptationAction
    target: str  # "experto|régimen|horizonte" (o feed/símbolo)
    reason: str
    prediction_id: str
    cause: ErrorCause

    def __post_init__(self) -> None:
        if not isinstance(self.action, AdaptationAction):
            raise TypeError("action debe ser AdaptationAction")
        if not isinstance(self.cause, ErrorCause):
            raise TypeError("cause debe ser ErrorCause")
        for name in ("target", "reason", "prediction_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} no puede ser vacío")


def propose_directive(
    attribution: ErrorAttribution,
    *,
    expert: str,
    regime: str,
    horizon_seconds: int,
    prediction_id: str,
) -> AdaptationDirective:
    """Propone la directiva para la causa primaria (puro)."""
    if not isinstance(attribution, ErrorAttribution):
        raise TypeError("attribution debe ser ErrorAttribution")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds debe ser > 0")
    action = _CAUSE_TO_ACTION[attribution.primary]
    return AdaptationDirective(
        action=action,
        target=f"{expert.strip()}|{regime.strip()}|{horizon_seconds}s",
        reason=f"{attribution.primary.value}: {attribution.detail}",
        prediction_id=prediction_id.strip(),
        cause=attribution.primary,
    )
