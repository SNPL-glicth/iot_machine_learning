"""Trading MVP 0.1 — Evidence Gate: decisión paper NO_TRADE/LONG/SHORT.

Dominio puro: consume ``CalibrationEvidence`` y produce una decisión
ejecutable en papel. Reglas de honestidad:

- Sin calibrador disponible (UNCALIBRATED) ⇒ NO_TRADE. Nunca se opera con
  confianza inventada.
- Zona neutral alrededor de 0.5 ⇒ NO_TRADE: la ventaja bruta no cubre
  costes/incertidumbre.
- LONG solo con prob_calibrated >= 0.5 + margen; SHORT con <= 0.5 - margen.

La decisión se registra junto a la evidencia: el experimento paper debe ser
reproducible (NO-TRADE RATE es una de las métricas del MVP).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .pipeline import UNCALIBRATED, CalibrationEvidence

__all__ = [
    "TradeAction",
    "GateReason",
    "PaperDecision",
    "EvidenceRecord",
    "EvidenceGate",
]


class TradeAction(str, Enum):
    """Acción de papel (sin dinero real)."""

    NO_TRADE = "NO_TRADE"
    LONG = "LONG"
    SHORT = "SHORT"


class GateReason(str, Enum):
    """Por qué el gate produjo la acción (auditoria del experimento)."""

    UNCALIBRATED = "uncalibrated"
    NEUTRAL_ZONE = "neutral_zone"
    LONG_SIGNAL = "long_signal"
    SHORT_SIGNAL = "short_signal"


@dataclass(frozen=True)
class PaperDecision:
    """Decisión del gate sobre una señal emitida."""

    action: TradeAction
    reason: GateReason
    probability: float


@dataclass(frozen=True)
class EvidenceRecord:
    """Evidencia de calibración + decisión del gate para una señal.

    Es la unidad que se persiste: convierte el paper bot en experimento
    reproducible (qué creyó el modelo, qué corrigió el calibrador, qué
    habría operado).
    """

    evidence: CalibrationEvidence
    decision: PaperDecision


class EvidenceGate:
    """Umbral de evidencia para operar en papel.

    Args:
        neutral_margin: semi-ancho de la zona neutral alrededor de 0.5
            (0.05 ⇒ opera LONG con p >= 0.55). Debe estar en [0, 0.45].
        require_calibrated: si True (default), UNCALIBRATED ⇒ NO_TRADE.
            Desactivarlo solo para experimentos controlados con el predictor
            crudo; el motivo queda igualmente registrado.
    """

    def __init__(
        self,
        *,
        neutral_margin: float = 0.05,
        require_calibrated: bool = True,
    ) -> None:
        if not 0.0 <= neutral_margin <= 0.45:
            raise ValueError(
                f"neutral_margin fuera de [0, 0.45]: {neutral_margin!r}"
            )
        self.neutral_margin = neutral_margin
        self.require_calibrated = require_calibrated

    def decide(self, evidence: CalibrationEvidence) -> PaperDecision:
        """Decide la acción de papel para una señal calibrada."""
        probability = evidence.prob_calibrated
        if self.require_calibrated and evidence.fallback_level == UNCALIBRATED:
            return PaperDecision(
                TradeAction.NO_TRADE, GateReason.UNCALIBRATED, probability
            )
        upper = 0.5 + self.neutral_margin
        lower = 0.5 - self.neutral_margin
        if probability >= upper:
            return PaperDecision(
                TradeAction.LONG, GateReason.LONG_SIGNAL, probability
            )
        if probability <= lower:
            return PaperDecision(
                TradeAction.SHORT, GateReason.SHORT_SIGNAL, probability
            )
        return PaperDecision(
            TradeAction.NO_TRADE, GateReason.NEUTRAL_ZONE, probability
        )
