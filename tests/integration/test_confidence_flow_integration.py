"""Integration test for full confidence calibration flow.

Tests the complete pipeline:
1. Raw confidence from engines
2. Domain penalty-based calibration (pre-decision)
3. Fusion (weighted average)
4. Infrastructure temperature scaling (post-fusion)
5. Final scaled confidence

This test uses mocked engines to isolate confidence calibration logic.
"""

import pytest

from core.tuning.temperature_scaling import TemperatureScaler
from domain.services.confidence_calibrator import ConfidenceCalibrator


class TestConfidenceFlowIntegration:
    """Integration tests for complete confidence calibration flow."""

    def test_happy_path_full_flow(self):
        """Happy path: raw confidence → domain calibrate → fuse → scale."""
        # Step 1: Raw confidence from engines (simulated)
        raw_engine_confidences = [0.85, 0.80, 0.90]
        fused_confidence = sum(raw_engine_confidences) / len(raw_engine_confidences)  # 0.85

        # Step 2: Domain penalty-based calibration (pre-decision)
        domain_calibrator = ConfidenceCalibrator()
        domain_result = domain_calibrator.calibrate(
            raw_confidence=fused_confidence,
            n_points=20,  # Good sample size
            noise_ratio=0.3,  # Low noise
            engine_disagreement=0.1,  # Low disagreement
            only_baseline_active=False,
            coherence_conflict=False,
            all_engines_inhibited=False,
        )
        # Should have minimal penalties
        assert domain_result.calibrated >= 0.70  # Some penalty may apply
        assert domain_result.penalty_applied >= 0.0

        # Step 3: Use calibrated confidence for decision (simulated)
        # In real flow, this would go to decision engine
        decision_confidence = domain_result.calibrated

        # Step 4: Infrastructure temperature scaling (post-fusion)
        # In real flow, this happens in ConfidenceCalibrationPhase
        temp_scaler = TemperatureScaler()
        temp_result = temp_scaler.scale(
            confidence=decision_confidence,
            regime="STABLE",
        )

        # Step 5: Verify final scaled confidence
        assert 0.3 <= temp_result.scaled_confidence <= 0.95  # Floor/ceiling
        assert temp_result.temperature_used == 1.2  # STABLE regime
        assert "sigmoid" in temp_result.formula

    def test_edge_case_low_quality_data(self):
        """Edge case: low quality data triggers penalties and scaling."""
        # Step 1: Raw confidence from engines (low)
        raw_engine_confidences = [0.60, 0.55, 0.65]
        fused_confidence = sum(raw_engine_confidences) / len(raw_engine_confidences)  # 0.60

        # Step 2: Domain penalty-based calibration with many penalties
        domain_calibrator = ConfidenceCalibrator()
        domain_result = domain_calibrator.calibrate(
            raw_confidence=fused_confidence,
            n_points=5,  # Low sample size
            noise_ratio=0.8,  # High noise
            engine_disagreement=0.5,  # High disagreement
            only_baseline_active=True,  # Only baseline
            coherence_conflict=True,  # Coherence conflict
            all_engines_inhibited=False,
        )

        # Should have significant penalties, but capped at 50%
        assert domain_result.penalty_applied > 0.20
        assert domain_result.calibrated >= 0.05  # Floor
        assert "penalty_capped" in domain_result.reasons[-1] or domain_result.penalty_applied <= 0.30

        # Step 3: Temperature scaling with VOLATILE regime
        temp_scaler = TemperatureScaler()
        temp_result = temp_scaler.scale(
            confidence=domain_result.calibrated,
            regime="VOLATILE",
        )

        # VOLATILE regime should further reduce confidence
        assert temp_result.temperature_used == 2.0
        assert temp_result.scaled_confidence >= 0.3  # Floor from CONFIDENCE.MIN

    def test_edge_case_high_confidence_with_consensus(self):
        """Edge case: high confidence with strong consensus."""
        # Step 1: Raw confidence from engines (high)
        raw_engine_confidences = [0.95, 0.94, 0.96]
        fused_confidence = sum(raw_engine_confidences) / len(raw_engine_confidences)  # 0.95

        # Step 2: Domain penalty-based calibration (minimal penalties)
        domain_calibrator = ConfidenceCalibrator()
        domain_result = domain_calibrator.calibrate(
            raw_confidence=fused_confidence,
            n_points=50,  # Large sample
            noise_ratio=0.1,  # Very low noise
            engine_disagreement=0.05,  # Very low disagreement
            only_baseline_active=False,
            coherence_conflict=False,
            all_engines_inhibited=False,
        )

        # Should have minimal penalties
        assert domain_result.penalty_applied < 0.10
        assert domain_result.calibrated > 0.85

        # Step 3: Temperature scaling with STABLE regime
        temp_scaler = TemperatureScaler()
        temp_result = temp_scaler.scale(
            confidence=domain_result.calibrated,
            regime="STABLE",
        )

        # Temperature scaling centers around 0.5, so high confidence gets reduced
        # This is expected behavior - sigmoid((c - 0.5) / T)
        assert 0.5 <= temp_result.scaled_confidence <= 0.95  # Valid range
        assert temp_result.temperature_used == 1.2
        # The scaling is correct: sigmoid centers around 0.5
        assert temp_result.scaled_confidence > 0.5  # Above center

    def test_floor_respected_in_full_flow(self):
        """Verify floor is respected throughout the flow."""
        # Step 1: Very low raw confidence
        fused_confidence = 0.20

        # Step 2: Domain calibration with penalties
        domain_calibrator = ConfidenceCalibrator()
        domain_result = domain_calibrator.calibrate(
            raw_confidence=fused_confidence,
            n_points=3,
            noise_ratio=0.9,
            engine_disagreement=0.8,
            only_baseline_active=True,
            coherence_conflict=True,
            all_engines_inhibited=False,
        )

        # Should hit floor 0.05
        assert domain_result.calibrated >= 0.05

        # Step 3: Temperature scaling
        temp_scaler = TemperatureScaler()
        temp_result = temp_scaler.scale(
            confidence=domain_result.calibrated,
            regime="NOISY",
        )

        # Should respect floor 0.3 from CONFIDENCE.MIN
        assert temp_result.scaled_confidence >= 0.3
