"""Trading MVP 0.1 — Evidence Gate y artefacto de calibrador versionable."""

from __future__ import annotations

import json
import random

import pytest

from iot_machine_learning.domain.entities.market.calibration import (
    AdaptiveCalibrator,
    CalibrationEvidence,
    CalibrationMethod,
    ContextKey,
    EvidenceGate,
    GateReason,
    TradeAction,
    export_calibrator_state,
    import_calibrator_state,
    try_refit,
)
from iot_machine_learning.domain.entities.market.calibration.pipeline import (
    UNCALIBRATED,
)


def _evidence(
    prob_calibrated: float,
    fallback_level: str = "context",
    version: str | None = "v1",
) -> CalibrationEvidence:
    return CalibrationEvidence(
        prediction_id="BTC-USD-momentum-3600-900",
        symbol="BTC-USD",
        horizon_seconds=900,
        regime="ALL",
        prob_raw=0.90,
        prob_calibrated=prob_calibrated,
        fallback_level=fallback_level,
        calibrator_version=version,
        observation_timestamp=3600.0,
    )


# ─── EvidenceGate ───────────────────────────────────────────────────────────


class TestEvidenceGate:
    def test_uncalibrated_implica_no_trade(self):
        decision = EvidenceGate().decide(_evidence(0.99, fallback_level=UNCALIBRATED))
        assert decision.action is TradeAction.NO_TRADE
        assert decision.reason is GateReason.UNCALIBRATED

    def test_matriz_long_short_neutral(self):
        gate = EvidenceGate(neutral_margin=0.05)
        assert gate.decide(_evidence(0.80)).action is TradeAction.LONG
        assert gate.decide(_evidence(0.55)).action is TradeAction.LONG
        assert gate.decide(_evidence(0.20)).action is TradeAction.SHORT
        assert gate.decide(_evidence(0.45)).action is TradeAction.SHORT
        neutral = gate.decide(_evidence(0.52))
        assert neutral.action is TradeAction.NO_TRADE
        assert neutral.reason is GateReason.NEUTRAL_ZONE
        # Fronteras inclusivas: 0.55 entra LONG, 0.45 entra SHORT.
        assert gate.decide(_evidence(0.5499)).action is TradeAction.NO_TRADE

    def test_margen_cero_opera_con_cualquier_sesion(self):
        gate = EvidenceGate(neutral_margin=0.0)
        assert gate.decide(_evidence(0.50)).action is TradeAction.LONG
        assert gate.decide(_evidence(0.4999)).action is TradeAction.SHORT

    def test_margen_invalido_rechazado(self):
        with pytest.raises(ValueError):
            EvidenceGate(neutral_margin=0.6)

    def test_require_calibrated_false_permite_raw_con_registro(self):
        gate = EvidenceGate(require_calibrated=False, neutral_margin=0.05)
        decision = gate.decide(_evidence(0.90, fallback_level=UNCALIBRATED))
        assert decision.action is TradeAction.LONG


# ─── Artefacto de calibrador ────────────────────────────────────────────────


def _fitted() -> AdaptiveCalibrator:
    rng = random.Random(7)
    ctx = ContextKey("momentum", 900, "ALL")
    pairs = [(ctx, 0.90, rng.random() < 0.50) for _ in range(600)]
    calibrator = AdaptiveCalibrator(method=CalibrationMethod.PLATT)
    calibrator.set_version("v4")
    assert try_refit(calibrator, pairs) is True
    return calibrator


class TestCalibratorArtifact:
    def test_round_trip_json_preserva_version_y_niveles(self):
        original = _fitted()
        state = export_calibrator_state(original)
        payload = json.loads(json.dumps(state))  # JSON-safe estricto

        restored = import_calibrator_state(payload)
        assert restored.get_version() == "v4"
        ctx = ContextKey("momentum", 900, "ALL")
        for prob in (0.30, 0.55, 0.90):
            a = original.apply_with_fallback(ctx, prob).prob_calibrated
            b = restored.apply_with_fallback(ctx, prob).prob_calibrated
            assert abs(a - b) < 1e-12

    def test_artefacto_incompatible_rechazado(self):
        with pytest.raises(ValueError, match="artifact_version"):
            import_calibrator_state({"artifact_version": 99})

    def test_artefacto_sin_calibradores_importa_passthrough(self):
        empty = AdaptiveCalibrator()
        restored = import_calibrator_state(export_calibrator_state(empty))
        result = restored.apply_with_fallback(
            ContextKey("momentum", 900, "ALL"), 0.9
        )
        assert result.is_available is False
        assert result.prob_calibrated == 0.9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
