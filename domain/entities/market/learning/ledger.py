"""Ledger de aprendizaje (FASE 7) — la memoria de los errores.

Append-only en memoria: cada error atribuido propone una directiva;
el refit offline las acepta (marca aplicada con versión) o las deja
pendientes. Estado JSON ``learning-ledger-v1`` para el artefacto
versionable (carga en arranque, como el calibrador y el meta-learner).

Sin aprendizaje en vivo: el ledger propone, no toca modelos.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .directives import AdaptationAction, AdaptationDirective
from .error_taxonomy import ErrorAttribution, ErrorCause

__all__ = ["LearningRecord", "LearningLedger", "LEDGER_STATE_VERSION"]

LEDGER_STATE_VERSION: str = "learning-ledger-v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class LearningRecord:
    """Un error aprendido (propuesta + estado de aplicación)."""

    record_id: str
    prediction_id: str
    cause: ErrorCause
    action: AdaptationAction
    target: str
    reason: str
    applied: bool = False
    applied_version: str | None = None


class LearningLedger:
    """Libro append-only de errores y directivas (en memoria)."""

    def __init__(self) -> None:
        self._records: dict[str, LearningRecord] = {}
        self._seq = 0

    def propose(
        self, attribution: ErrorAttribution, directive: AdaptationDirective
    ) -> LearningRecord:
        """Registra una propuesta (retorna el record)."""
        if not isinstance(attribution, ErrorAttribution):
            raise TypeError("attribution debe ser ErrorAttribution")
        if not isinstance(directive, AdaptationDirective):
            raise TypeError("directive debe ser AdaptationDirective")
        if directive.cause is not attribution.primary:
            raise ValueError("directive.cause debe ser attribution.primary")
        self._seq += 1
        record = LearningRecord(
            record_id=f"learn-{self._seq:06d}",
            prediction_id=directive.prediction_id,
            cause=attribution.primary,
            action=directive.action,
            target=directive.target,
            reason=directive.reason,
        )
        self._records[record.record_id] = record
        return record

    def mark_applied(self, record_id: str, version: str) -> LearningRecord:
        """Marca una propuesta como aplicada por una versión offline."""
        record = self._records.get(record_id)
        if record is None:
            raise KeyError(f"record desconocido: {record_id!r}")
        if not version.strip():
            raise ValueError("version no puede ser vacía")
        updated = LearningRecord(
            record_id=record.record_id,
            prediction_id=record.prediction_id,
            cause=record.cause,
            action=record.action,
            target=record.target,
            reason=record.reason,
            applied=True,
            applied_version=version.strip(),
        )
        self._records[record_id] = updated
        return updated

    def pending(self) -> tuple[LearningRecord, ...]:
        """Propuestas sin aplicar (la cola del refit offline)."""
        return tuple(r for r in self._records.values() if not r.applied)

    def summary(self) -> dict[str, dict[str, int]]:
        """Conteos por causa y acción + tasa de aplicación."""
        by_cause = Counter(r.cause.value for r in self._records.values())
        by_action = Counter(r.action.value for r in self._records.values())
        applied = sum(1 for r in self._records.values() if r.applied)
        return {
            "by_cause": dict(by_cause),
            "by_action": dict(by_action),
            "total": len(self._records),
            "applied": applied,
            "pending": len(self._records) - applied,
        }

    def to_state(self) -> dict:
        """Estado serializable (JSON) con versión."""
        return {
            "version": LEDGER_STATE_VERSION,
            "seq": self._seq,
            "records": [
                {
                    "record_id": r.record_id,
                    "prediction_id": r.prediction_id,
                    "cause": r.cause.value,
                    "action": r.action.value,
                    "target": r.target,
                    "reason": r.reason,
                    "applied": r.applied,
                    "applied_version": r.applied_version,
                }
                for r in self._records.values()
            ],
        }

    @classmethod
    def from_state(cls, state: dict) -> LearningLedger:
        """Restaura desde ``to_state`` (versión estricta)."""
        if state.get("version") != LEDGER_STATE_VERSION:
            raise ValueError(
                f"versión de estado desconocida: {state.get('version')!r}"
            )
        ledger = cls()
        for item in state.get("records", []):
            record = LearningRecord(
                record_id=item["record_id"],
                prediction_id=item["prediction_id"],
                cause=ErrorCause(item["cause"]),
                action=AdaptationAction(item["action"]),
                target=item["target"],
                reason=item["reason"],
                applied=bool(item.get("applied", False)),
                applied_version=item.get("applied_version"),
            )
            ledger._records[record.record_id] = record
        ledger._seq = int(state.get("seq", len(ledger._records)))
        return ledger
