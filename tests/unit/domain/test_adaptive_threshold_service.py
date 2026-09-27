"""Unit tests for AdaptiveThresholdService and calibration entities.

Conforms to:
- ISO/IEC 25010:2023: Reliability, fault tolerance, functional suitability.
- ISO/IEC 22989:2022: Continuous learning calibration and auditability.
"""

from __future__ import annotations

import pytest
from domain.entities.calibration.threshold_state import AdaptiveThresholdState
from domain.services.calibration.adaptive_threshold_service import AdaptiveThresholdService


def test_adaptive_threshold_service_initialization_invariants() -> None:
    """Verify service constructs valid baseline state satisfying invariant bounds."""
    service = AdaptiveThresholdService(
        decay_factor=0.95, default_multiplier_k=2.5, default_floor=1.5, default_ceiling=6.0
    )
    state = service.initialize_state(initial_mean=2.0, initial_variance=0.25)

    assert state.sample_count == 1
    assert state.running_mean == 2.0
    assert state.running_variance == 0.25
    assert state.floor_threshold == 1.5
    assert state.ceiling_threshold == 6.0
    assert state.floor_threshold <= state.current_threshold <= state.ceiling_threshold


def test_adaptive_threshold_entity_validation() -> None:
    """Verify invariant validation errors in AdaptiveThresholdState."""
    with pytest.raises(ValueError, match="sample_count must be non-negative"):
        AdaptiveThresholdState(
            sample_count=-1, running_mean=1.0, running_variance=0.1,
            current_threshold=2.0, floor_threshold=1.0, ceiling_threshold=5.0,
            multiplier_k=2.0, last_update_timestamp_ns=0,
        )

    with pytest.raises(ValueError, match="ceiling_threshold .* cannot be less than"):
        AdaptiveThresholdState(
            sample_count=1, running_mean=1.0, running_variance=0.1,
            current_threshold=2.0, floor_threshold=5.0, ceiling_threshold=2.0,
            multiplier_k=2.0, last_update_timestamp_ns=0,
        )


def test_adaptive_threshold_update_adapts_to_shocks_and_respects_ceiling() -> None:
    """Verify online moment update raises threshold under higher volatility up to ceiling."""
    service = AdaptiveThresholdService(
        decay_factor=0.90, default_multiplier_k=2.0, default_floor=1.5, default_ceiling=5.0
    )
    state = service.initialize_state(initial_mean=2.0, initial_variance=0.1)

    initial_th = state.current_threshold

    # Ingest multiple elevated values
    for _ in range(10):
        state = service.update_threshold(observed_metric=4.5, state=state)

    assert state.sample_count == 11
    assert state.running_mean > 2.0
    assert state.current_threshold > initial_th
    assert state.current_threshold <= 5.0  # Ceiling respected


def test_adaptive_threshold_respects_floor_under_low_dispersion() -> None:
    """Verify floor threshold prevents unsafe collapse of threshold under zero dispersion."""
    service = AdaptiveThresholdService(
        decay_factor=0.90, default_multiplier_k=2.0, default_floor=1.8, default_ceiling=6.0
    )
    state = service.initialize_state(initial_mean=0.1, initial_variance=0.0001)

    # Ingest very small values
    for _ in range(20):
        state = service.update_threshold(observed_metric=0.05, state=state)

    assert state.current_threshold >= 1.8  # Floor strictly guaranteed


def test_evaluate_admissibility_snapshot() -> None:
    """Verify audit snapshot outputs deterministic booleans and telemetry."""
    service = AdaptiveThresholdService()
    state = service.initialize_state(initial_mean=2.0, initial_variance=0.25)

    snap_pass = service.evaluate_admissibility(observed_value=2.2, state=state)
    assert snap_pass.is_admissible is True
    assert snap_pass.observed_value == 2.2

    snap_fail = service.evaluate_admissibility(observed_value=5.5, state=state)
    assert snap_fail.is_admissible is False


def test_compute_adaptive_confidence_bands() -> None:
    """Verify dynamic confidence bands scale monotonically with base dispersion."""
    service = AdaptiveThresholdService()

    bands_low = service.compute_adaptive_confidence_bands(base_dispersion=0.002)
    bands_high = service.compute_adaptive_confidence_bands(base_dispersion=0.010)

    assert len(bands_low) == 5
    assert len(bands_high) == 5
    # High dispersion must widen the target ratio
    assert bands_high[0].target_ratio > bands_low[0].target_ratio
    assert bands_high[0].stop_ratio > bands_low[0].stop_ratio
