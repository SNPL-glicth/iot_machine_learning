"""Trading MVP 0.1 — Mapeos del repositorio de evidencia (sin MySQL)."""

from __future__ import annotations

import json

from iot_machine_learning.domain.entities.market.calibration import (
    CalibrationEvidence,
    EvidenceGate,
    GateReason,
    TradeAction,
)
from iot_machine_learning.domain.entities.market.calibration.pipeline import (
    UNCALIBRATED,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.calibration_evidence_repository import (
    EvidenceRecord,
    record_to_row,
    row_to_record,
)


def _record() -> EvidenceRecord:
    evidence = CalibrationEvidence(
        prediction_id="BTC-USD-momentum-1748000000-900",
        symbol="BTC-USD",
        horizon_seconds=900,
        regime="ALL",
        prob_raw=0.90,
        prob_calibrated=0.51,
        fallback_level=UNCALIBRATED,
        calibrator_version="v4",
        observation_timestamp=1_748_000_000.0,
    )
    decision = EvidenceGate().decide(evidence)
    return EvidenceRecord(evidence=evidence, decision=decision)


class TestRowRoundTrip:
    def test_record_a_fila_a_registro_sin_perdida(self):
        original = _record()
        row = record_to_row(original)
        restored = row_to_record(row)

        assert restored.evidence.prediction_id == original.evidence.prediction_id
        assert restored.evidence.prob_raw == 0.90
        assert restored.evidence.prob_calibrated == 0.51
        assert restored.evidence.fallback_level == UNCALIBRATED
        assert restored.evidence.calibrator_version == "v4"
        assert restored.decision.action is TradeAction.NO_TRADE
        assert restored.decision.reason is GateReason.UNCALIBRATED

    def test_fila_json_safe(self):
        row = record_to_row(_record())
        payload = json.loads(json.dumps(row, default=str))
        assert payload["paper_action"] == "NO_TRADE"

    def test_long_signal_fluye_completo(self):
        evidence = CalibrationEvidence(
            prediction_id="x-900",
            symbol="BTC-USD",
            horizon_seconds=900,
            regime="bull",
            prob_raw=0.60,
            prob_calibrated=0.72,
            fallback_level="context",
            calibrator_version="v4",
            observation_timestamp=1_748_000_060.0,
        )
        record = EvidenceRecord(
            evidence=evidence, decision=EvidenceGate().decide(evidence)
        )
        restored = row_to_record(record_to_row(record))
        assert restored.decision.action is TradeAction.LONG
        assert restored.decision.reason is GateReason.LONG_SIGNAL


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))
