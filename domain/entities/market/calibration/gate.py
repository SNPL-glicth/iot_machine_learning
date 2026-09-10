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
from ..costs import CostModel
from ..costs_net import evaluate_net

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
    DISTRIBUTION_SHIFT = "distribution_shift"  # FASE 4: anomalía veta
    GROSS_BLOCKED = "gross_blocked"  # FASE 6: ni en bruto hay edge
    COST_BLOCKED = "cost_blocked"  # FASE 6: los costos matan la señal


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
        return self.decide_with_shift(evidence, is_shift=False)

    def decide_with_costs(
        self,
        evidence: CalibrationEvidence,
        *,
        expected_gross: float,
        cost_model: CostModel,
    ) -> PaperDecision:
        """Decide con gate económico duro (FASE 6).

        Orden de vetos: UNCALIBRATED → SHIFT n/a aquí → GROSS (bruto<=0)
        → COST (neto<=0) → probabilidad. Un 0.80 calibrado con +0.10%
        bruto en BTC (24bps) da NO_TRADE/COST_BLOCKED: exacto, la señal
        no paga su intento.         La paridad con la escalera de ``costs`` es
        total: gross_negative→GROSS_BLOCKED, cost_negative→COST_BLOCKED.
        """
        probability = evidence.prob_calibrated
        if self.require_calibrated and evidence.fallback_level == UNCALIBRATED:
            return PaperDecision(
                TradeAction.NO_TRADE, GateReason.UNCALIBRATED, probability
            )
        evaluation = evaluate_net(expected_gross, cost_model)
        if evaluation.gross_return <= 0:
            return PaperDecision(
                TradeAction.NO_TRADE, GateReason.GROSS_BLOCKED, probability
            )
        if evaluation.net_return <= 0:
            return PaperDecision(
                TradeAction.NO_TRADE, GateReason.COST_BLOCKED, probability
            )
        return self._decide_probability(probability)

    def _decide_probability(self, probability: float) -> PaperDecision:
        """LONG/SHORT/NEUTRAL puro sobre la probabilidad calibrada."""
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

    def decide_with_shift(
        self, evidence: CalibrationEvidence, is_shift: bool = False,
    ) -> PaperDecision:
        """Decide con veto de anomalía (FASE 4).

        SHIFT ⇒ NO_TRADE antes de mirar la señal: evitar operar también
        es una decisión y queda registrada con su motivo. Sin shift,
        idéntico a ``decide`` (compatibilidad total).
        """
        probability = evidence.prob_calibrated
        if is_shift:
            return PaperDecision(
                TradeAction.NO_TRADE, GateReason.DISTRIBUTION_SHIFT, probability
            )
        if self.require_calibrated and evidence.fallback_level == UNCALIBRATED:
            return PaperDecision(
                TradeAction.NO_TRADE, GateReason.UNCALIBRATED, probability
            )
        return self._decide_probability(probability)
