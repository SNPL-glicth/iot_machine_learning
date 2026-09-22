"""Unit tests for CalibrationService and CalibratorRepositoryPort."""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.entities.market.calibration import FallbackLevel
from iot_machine_learning.domain.ports.calibrator_repository_port import (
    CalibratorRepositoryPort,
    InMemoryCalibratorRepository,
)
from iot_machine_learning.domain.services.calibration_service import (
    CalibratedPrediction,
    CalibrationContext,
    CalibrationService,
    EvidenceGateDecision,
)


def test_calibration_service_initialization_in_memory() -> None:
    """CalibrationService can be initialized with default in-memory repository."""
    service = CalibrationService()
    assert isinstance(service._repo, CalibratorRepositoryPort)
    assert not service.load_active_calibrator()


def test_calibration_service_apply_calibration_without_active_calibrator() -> None:
    """When no calibrator is loaded, evidence gate decision follows require_calibration."""
    # With require_calibration=True -> NO_TRADE
    service_strict = CalibrationService(require_calibration=True)
    ctx = CalibrationContext(
        symbol="BTC-USD",
        strategy="momentum",
        horizon_seconds=3600,
        regime="BULL",
        model_version="v1",
        strategy_version="v1",
        evidence_policy_version="v1",
    )
    result = service_strict.apply_calibration(ctx, prob_raw=0.6)
    assert result.evidence_gate_decision == EvidenceGateDecision.NO_TRADE
    assert result.fallback_level == FallbackLevel.UNAVAILABLE
    assert not result.calibration_applied

    # With require_calibration=False -> RAW
    service_permissive = CalibrationService(require_calibration=False)
    result2 = service_permissive.apply_calibration(ctx, prob_raw=0.6)
    assert result2.evidence_gate_decision == EvidenceGateDecision.RAW
    assert result2.prob_calibrated == 0.6
