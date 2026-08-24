"""CalibrationEvidenceRepository (Trading MVP 0.1) — evidencia 10.5 en MySQL.

Persiste ``CalibrationEvidence`` + ``PaperDecision`` por señal emitida en
vivo. Append-only, upsert idempotente por prediction_id (re-correr un ciclo
jamás duplica filas). Responde: "¿con qué calibrador trabajaba ZENIN a las
14:37 y qué habría operado?".

Las funciones de mapeo fila <-> registro son puras para cubrirlas con
tests sin MySQL.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from iot_machine_learning.domain.entities.market.calibration.gate import (
    EvidenceRecord,
    GateReason,
    PaperDecision,
    TradeAction,
)
from iot_machine_learning.domain.entities.market.calibration.pipeline import (
    CalibrationEvidence,
)

__all__ = ["record_to_row", "row_to_record", "CalibrationEvidenceRepository"]


_UPSERT_SQL = text(
    """
    INSERT INTO calibration_evidence (
        prediction_id, symbol, horizon_seconds, observation_timestamp,
        regime, prob_raw, prob_calibrated, fallback_level,
        calibrator_version, paper_action, gate_reason
    ) VALUES (
        :prediction_id, :symbol, :horizon_seconds, :observation_timestamp,
        :regime, :prob_raw, :prob_calibrated, :fallback_level,
        :calibrator_version, :paper_action, :gate_reason
    )
    ON DUPLICATE KEY UPDATE
        prob_raw = VALUES(prob_raw),
        prob_calibrated = VALUES(prob_calibrated),
        fallback_level = VALUES(fallback_level),
        calibrator_version = VALUES(calibrator_version),
        paper_action = VALUES(paper_action),
        gate_reason = VALUES(gate_reason)
    """
)

_SELECT_SQL = text(
    """
    SELECT prediction_id, symbol, horizon_seconds, observation_timestamp,
           regime, prob_raw, prob_calibrated, fallback_level,
           calibrator_version, paper_action, gate_reason
    FROM calibration_evidence
    """
)


def record_to_row(record: EvidenceRecord) -> dict[str, Any]:
    """Registro → fila MySQL (puro)."""
    ev, dec = record.evidence, record.decision
    return {
        "prediction_id": ev.prediction_id,
        "symbol": ev.symbol,
        "horizon_seconds": int(ev.horizon_seconds),
        "observation_timestamp": float(ev.observation_timestamp),
        "regime": ev.regime,
        "prob_raw": float(ev.prob_raw),
        "prob_calibrated": float(ev.prob_calibrated),
        "fallback_level": ev.fallback_level,
        "calibrator_version": ev.calibrator_version,
        "paper_action": dec.action.value,
        "gate_reason": dec.reason.value,
    }


def row_to_record(row: Any) -> EvidenceRecord:
    """Fila MySQL → registro (puro; acepta dict o Row con keys)."""
    get = row._mapping if hasattr(row, "_mapping") else row  # noqa: SLF001
    return EvidenceRecord(
        evidence=CalibrationEvidence(
            prediction_id=get["prediction_id"],
            symbol=get["symbol"],
            horizon_seconds=int(get["horizon_seconds"]),
            regime=get["regime"],
            prob_raw=float(get["prob_raw"]),
            prob_calibrated=float(get["prob_calibrated"]),
            fallback_level=get["fallback_level"],
            calibrator_version=get["calibrator_version"],
            observation_timestamp=float(get["observation_timestamp"]),
        ),
        decision=PaperDecision(
            action=TradeAction(get["paper_action"]),
            reason=GateReason(get["gate_reason"]),
            probability=float(get["prob_calibrated"]),
        ),
    )


class CalibrationEvidenceRepository:
    """Acceso a ``calibration_evidence`` (solo guardar y consultar)."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def save_batch(self, records: Iterable[EvidenceRecord]) -> int:
        """Upsert idempotente de un lote; retorna filas escritas."""
        rows = [record_to_row(r) for r in records]
        for row in rows:
            self._conn.execute(_UPSERT_SQL, row)
        return len(rows)

    def recent(self, *, symbol: str, limit: int = 20) -> list[EvidenceRecord]:
        """Últimas señales emitidas (más nueva primero)."""
        result = self._conn.execute(
            _SELECT_SQL
            + " WHERE symbol = :symbol ORDER BY observation_timestamp DESC LIMIT :limit",
            {"symbol": symbol, "limit": int(limit)},
        )
        return [row_to_record(r) for r in result]

    def at_timestamp(
        self, symbol: str, timestamp: float, tolerance_seconds: float = 120.0
    ) -> list[EvidenceRecord]:
        """Evidencia alrededor de un instante (la pregunta de las 14:37)."""
        result = self._conn.execute(
            _SELECT_SQL
            + """
            WHERE symbol = :symbol
              AND observation_timestamp BETWEEN :lo AND :hi
            ORDER BY horizon_seconds ASC
            """,
            {
                "symbol": symbol,
                "lo": timestamp - tolerance_seconds,
                "hi": timestamp + tolerance_seconds,
            },
        )
        return [row_to_record(r) for r in result]

    def action_counts(self, symbol: str) -> dict[str, int]:
        """Conteo por acción (NO-TRADE RATE del status line)."""
        result = self._conn.execute(
            text(
                "SELECT paper_action, COUNT(*) AS n FROM calibration_evidence "
                "WHERE symbol = :symbol GROUP BY paper_action"
            ),
            {"symbol": symbol},
        )
        return {r[0]: int(r[1]) for r in result}
