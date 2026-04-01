"""Tests for ConfidenceCalibrator domain service.

Covers additive penalty calibration:
- Only baseline active → -0.25
- n_points < 10 → -0.20
- noise_ratio > 0.6 → -0.15
- engine_disagreement > 0.3 → -0.15
- coherence_conflict → -0.20
- Floor: never < 0.05
- Ceil: raw > 0.95 without consensus → 0.85
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass

from iot_machine_learning.domain.services.confidence_calibrator import (
    CalibratedConfidence,
    ConfidenceCalibrator,
)


# Mock perception for testing disagreement
@dataclass
class MockPerception:
    engine_name: str
    predicted_value: float


class TestCalibratorBasics:
    """Basic construction and sanity checks."""

    def test_calibrator_construction(self):
        """ConfidenceCalibrator can be instantiated."""
        calibrator = ConfidenceCalibrator()
        assert calibrator is not None
        assert calibrator.CONFIDENCE_FLOOR == 0.05

    def test_calibrated_confidence_dataclass(self):
        """CalibratedConfidence is a proper frozen dataclass."""
        result = CalibratedConfidence(
            calibrated=0.65,
            raw=0.85,
            penalty_applied=0.20,
            reasons=["reason1", "reason2"],
        )
        assert result.calibrated == 0.65
        assert result.raw == 0.85
        assert result.penalty_applied == 0.20
        assert len(result.reasons) == 2


class TestOnlyBaselinePenalty:
    """Penalty: -0.25 when only baseline active or all inhibited."""

    def test_only_baseline_active_penalty(self):
        """only_baseline_active=True → -0.25 penalty."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.80,
            n_points=20,
            only_baseline_active=True,
        )
        assert result.calibrated == pytest.approx(0.55)  # 0.80 - 0.25
        assert result.penalty_applied == 0.25
        assert "only_baseline" in result.reasons[0]

    def test_all_engines_inhibited_penalty(self):
        """all_engines_inhibited=True → -0.25 penalty."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.80,
            n_points=20,
            all_engines_inhibited=True,
        )
        assert result.calibrated == 0.55
        assert result.penalty_applied == 0.25


class TestLowSampleSizePenalty:
    """Penalty: -0.20 when n_points < 10."""

    def test_n_points_less_than_10_penalty(self):
        """n_points=5 → -0.20 penalty."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.80,
            n_points=5,
        )
        assert result.calibrated == pytest.approx(0.60)  # 0.80 - 0.20
        assert "n_points=5" in result.reasons[0]

    def test_n_points_9_penalty(self):
        """n_points=9 → -0.20 penalty."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.80,
            n_points=9,
        )
        assert result.calibrated == pytest.approx(0.60)

    def test_n_points_10_no_penalty(self):
        """n_points=10 → no penalty."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.80,
            n_points=10,
        )
        assert result.calibrated == pytest.approx(0.80)
        assert result.reasons == []


class TestHighNoisePenalty:
    """Penalty: -0.15 when noise_ratio > 0.6."""

    def test_noise_ratio_07_penalty(self):
        """noise_ratio=0.7 → -0.15 penalty."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.80,
            n_points=20,
            noise_ratio=0.7,
        )
        assert result.calibrated == pytest.approx(0.65)  # 0.80 - 0.15
        assert "noise_ratio" in result.reasons[0]

    def test_noise_ratio_06_no_penalty(self):
        """noise_ratio=0.6 → no penalty (must be > 0.6)."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.80,
            n_points=20,
            noise_ratio=0.6,
        )
        assert result.calibrated == pytest.approx(0.80)


class TestEngineDisagreementPenalty:
    """Penalty: -0.15 when engine_disagreement > 0.3."""

    def test_disagreement_05_penalty(self):
        """engine_disagreement=0.5 → -0.15 penalty."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.80,
            n_points=20,
            engine_disagreement=0.5,
        )
        assert result.calibrated == pytest.approx(0.65)
        assert "engine_disagreement" in result.reasons[0]

    def test_disagreement_03_no_penalty(self):
        """engine_disagreement=0.3 → no penalty (must be > 0.3)."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.80,
            n_points=20,
            engine_disagreement=0.3,
        )
        assert result.calibrated == pytest.approx(0.80)


class TestCoherenceConflictPenalty:
    """Penalty: -0.20 when coherence_conflict=True."""

    def test_coherence_conflict_penalty(self):
        """coherence_conflict=True → -0.20 penalty."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.80,
            n_points=20,
            coherence_conflict=True,
        )
        assert result.calibrated == pytest.approx(0.60)  # 0.80 - 0.20
        assert "coherence_conflict" in result.reasons[0]

    def test_no_coherence_conflict_no_penalty(self):
        """coherence_conflict=False → no penalty."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.80,
            n_points=20,
            coherence_conflict=False,
        )
        assert result.calibrated == pytest.approx(0.80)


class TestAdditivePenalties:
    """Multiple penalties accumulate additively."""

    def test_baseline_and_low_sample(self):
        """-0.25 (baseline) + -0.20 (n_points) = -0.45."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.80,
            n_points=5,
            only_baseline_active=True,
        )
        assert result.calibrated == pytest.approx(0.35)  # 0.80 - 0.45
        assert len(result.reasons) == 2
        assert result.penalty_applied == pytest.approx(0.45)

    def test_all_penalties_stacked(self):
        """All penalties together, but ceil overrides when raw >= 0.95."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.95,
            n_points=5,
            noise_ratio=0.7,
            engine_disagreement=0.5,
            only_baseline_active=True,
            coherence_conflict=True,
        )
        # Ceil applies because raw >= 0.95 and no consensus, overriding penalties
        assert result.calibrated == pytest.approx(0.85)  # Ceil value
        assert result.penalty_applied == pytest.approx(0.10)  # 0.95 - 0.85
        assert len(result.reasons) == 6  # 5 penalties + ceil


class TestConfidenceFloor:
    """Floor: never returns confidence < 0.05."""

    def test_floor_at_005(self):
        """Even with massive penalties, floor is 0.05."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.20,
            n_points=5,
            noise_ratio=0.9,
            engine_disagreement=0.8,
            only_baseline_active=True,
            coherence_conflict=True,
        )
        assert result.calibrated == 0.05  # Floor
        assert result.calibrated >= calibrator.CONFIDENCE_FLOOR

    def test_floor_with_zero_raw(self):
        """raw=0 → calibrated=0.05 (floor)."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.0,
            n_points=20,
        )
        assert result.calibrated == 0.05


class TestConfidenceCeil:
    """Ceil: raw > 0.95 without strong evidence → clipear a 0.85."""

    def test_ceil_without_consensus(self):
        """raw=0.97, no consensus indicators → clipea a 0.85."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.97,
            n_points=20,
            noise_ratio=0.7,  # High noise = no strong evidence
        )
        assert result.calibrated == pytest.approx(0.85)  # Ceil applied
        assert "ceil_applied" in result.reasons[-1]

    def test_no_ceil_with_consensus(self):
        """raw=0.97 with strong evidence → no ceil."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.97,
            n_points=20,
            noise_ratio=0.3,  # Low noise
            engine_disagreement=0.1,  # Low disagreement
            only_baseline_active=False,
            coherence_conflict=False,
        )
        assert result.calibrated == pytest.approx(0.97)  # No ceil
        assert "ceil" not in " ".join(result.reasons)

    def test_ceil_raw_at_095_exact(self):
        """raw=0.95 exactly, boundary case."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.95,
            n_points=20,
            noise_ratio=0.7,
        )
        # Should apply ceil because noise_ratio > 0.4
        assert result.calibrated == pytest.approx(0.85)


class TestNoPenalties:
    """Clean case: no penalties, calibrated == raw."""

    def test_no_penalties_clean_pass(self):
        """All conditions good → calibrated == raw."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.75,
            n_points=20,
            noise_ratio=0.3,
            engine_disagreement=0.1,
            only_baseline_active=False,
            coherence_conflict=False,
        )
        assert result.calibrated == pytest.approx(0.75)
        assert result.raw == 0.75
        assert result.penalty_applied == 0.0
        assert result.reasons == []


class TestEngineDisagreementComputation:
    """compute_engine_disagreement helper."""

    def test_disagreement_single_perception(self):
        """Single perception → disagreement=0."""
        calibrator = ConfidenceCalibrator()
        perceptions = [MockPerception("taylor", 25.0)]
        disagreement = calibrator.compute_engine_disagreement(perceptions)
        assert disagreement == 0.0

    def test_disagreement_multiple_consensus(self):
        """Multiple perceptions in agreement → low disagreement."""
        calibrator = ConfidenceCalibrator()
        perceptions = [
            MockPerception("taylor", 25.0),
            MockPerception("kalman", 25.5),
            MockPerception("baseline", 24.8),
        ]
        disagreement = calibrator.compute_engine_disagreement(perceptions)
        assert disagreement < 0.05  # Very low disagreement

    def test_disagreement_multiple_divergent(self):
        """Multiple perceptions diverging → high disagreement."""
        calibrator = ConfidenceCalibrator()
        perceptions = [
            MockPerception("taylor", 25.0),
            MockPerception("kalman", 50.0),
            MockPerception("baseline", 75.0),
        ]
        disagreement = calibrator.compute_engine_disagreement(perceptions)
        assert disagreement > 0.3  # High disagreement

    def test_disagreement_empty_list(self):
        """Empty perceptions list → disagreement=0."""
        calibrator = ConfidenceCalibrator()
        disagreement = calibrator.compute_engine_disagreement([])
        assert disagreement == 0.0

    def test_disagreement_none_values(self):
        """Perceptions with None predicted_value ignored."""
        calibrator = ConfidenceCalibrator()
        perceptions = [
            MockPerception("taylor", 25.0),
            MockPerception("kalman", None),  # type: ignore
        ]
        disagreement = calibrator.compute_engine_disagreement(perceptions)
        assert disagreement == 0.0  # Only one valid value


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_very_high_raw_with_penalties(self):
        """raw=0.99 with many penalties → respects floor."""
        calibrator = ConfidenceCalibrator()
        result = calibrator.calibrate(
            raw_confidence=0.99,
            n_points=3,
            noise_ratio=0.9,
            only_baseline_active=True,
        )
        # Should be: 0.99 - 0.25 - 0.20 - 0.15 = 0.39
        # But floor is 0.05
        assert result.calibrated >= 0.05
        assert result.calibrated <= 0.99

    def test_boundary_n_points_9_vs_10(self):
        """n_points=9 penalized, n_points=10 not penalized."""
        calibrator = ConfidenceCalibrator()
        result_9 = calibrator.calibrate(raw_confidence=0.80, n_points=9)
        result_10 = calibrator.calibrate(raw_confidence=0.80, n_points=10)
        assert result_9.calibrated < result_10.calibrated

    def test_boundary_noise_06_vs_07(self):
        """noise=0.6 not penalized, noise=0.7 penalized."""
        calibrator = ConfidenceCalibrator()
        result_06 = calibrator.calibrate(
            raw_confidence=0.80, n_points=20, noise_ratio=0.6
        )
        result_07 = calibrator.calibrate(
            raw_confidence=0.80, n_points=20, noise_ratio=0.7
        )
        assert result_07.calibrated < result_06.calibrated
