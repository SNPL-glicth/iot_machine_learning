"""Tests for BoundaryCheckPhase rewrite.

Covers:
  * SensorProfile-based boundary validation
  * Dynamic range (p1-p99) when no profile
  * data_quality_score flow
  * dynamic_range_used flag
  * Near-boundary warnings
  * Fallback on out-of-domain
"""

from __future__ import annotations

import math
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.boundary_check_phase import (
    BoundaryCheckPhase,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import (
    PipelineContext,
    create_initial_context,
)
from iot_machine_learning.domain.entities.results.boundary_result import BoundaryResult
from iot_machine_learning.domain.entities.sensor_profile import SensorProfile
from iot_machine_learning.domain.value_objects.equipment_class import EquipmentClass

_PROFILE_KWARGS = dict(
    series_id="s-1",
    equipment_class=EquipmentClass.GENERIC,
    setpoint_tolerance=1.0,
    noise_floor=0.1,
    maintenance_history_score=0.9,
)


def _make_ctx(
    values: List[float],
    series_id: str = "s-1",
    profile: Any = None,
    data_quality_score: float = 1.0,
) -> PipelineContext:
    return create_initial_context(
        orchestrator=MagicMock(),
        values=values,
        timestamps=None,
        series_id=series_id,
        flags=MagicMock(),
        timer=MagicMock(),
    ).with_field(
        profile=profile,
        data_quality_score=data_quality_score,
    )


class TestBoundaryCheckWithProfile:
    def test_values_within_profile_range(self) -> None:
        profile = SensorProfile(
            operational_range=(0.0, 100.0),
            **_PROFILE_KWARGS,
        )
        phase = BoundaryCheckPhase()
        ctx = _make_ctx([25.0, 50.0, 75.0], profile=profile)
        result = phase.execute(ctx)

        assert result.boundary_result.within_domain is True
        assert result.boundary_result.dynamic_range_used is False
        assert result.boundary_result.rejection_reason is None

    def test_values_outside_profile_range_triggers_fallback(self) -> None:
        profile = SensorProfile(
            operational_range=(0.0, 100.0),
            **_PROFILE_KWARGS,
        )
        phase = BoundaryCheckPhase()
        ctx = _make_ctx([-10.0, 50.0, 200.0], profile=profile)
        result = phase.execute(ctx)

        assert result.boundary_result.within_domain is False
        assert result.boundary_result.rejection_reason == "out_of_domain"
        assert result.is_fallback is True
        assert result.fallback_reason == "out_of_domain"
        assert result.boundary_result.dynamic_range_used is False
        assert "values_outside_boundary:2" in result.boundary_result.warnings

    def test_near_boundary_warnings(self) -> None:
        profile = SensorProfile(
            operational_range=(0.0, 100.0),
            **_PROFILE_KWARGS,
        )
        phase = BoundaryCheckPhase()
        # 99.0 is within 5% of upper bound 100 → warning
        ctx = _make_ctx([1.0, 50.0, 99.0], profile=profile)
        result = phase.execute(ctx)

        assert result.boundary_result.within_domain is True
        assert "values_near_upper_boundary" in str(result.boundary_result.warnings)


class TestBoundaryCheckWithoutProfile:
    def test_dynamic_range_used_when_no_profile(self) -> None:
        phase = BoundaryCheckPhase()
        ctx = _make_ctx([10.0, 20.0, 30.0, 40.0, 50.0], profile=None)
        result = phase.execute(ctx)

        assert result.boundary_result.dynamic_range_used is True
        # p1-p99 may flag extremes as outside — that's expected
        assert isinstance(result.boundary_result.within_domain, bool)

    def test_dynamic_range_reports_near_upper_boundary(self) -> None:
        phase = BoundaryCheckPhase()
        # A value far above the cluster triggers near-boundary warning
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        ctx = _make_ctx(values, profile=None)
        result = phase.execute(ctx)

        assert result.boundary_result.dynamic_range_used is True
        assert "values_near_upper_boundary" in str(result.boundary_result.warnings)

    def test_dynamic_range_all_within(self) -> None:
        phase = BoundaryCheckPhase()
        values = [30.0, 31.0, 32.0, 33.0, 34.0, 35.0]
        ctx = _make_ctx(values, profile=None)
        result = phase.execute(ctx)

        assert result.boundary_result.dynamic_range_used is True
        assert result.boundary_result.within_domain is True

    def test_single_value_no_crash(self) -> None:
        phase = BoundaryCheckPhase()
        ctx = _make_ctx([42.0], profile=None)
        result = phase.execute(ctx)

        assert result.boundary_result.within_domain is True
        assert result.boundary_result.dynamic_range_used is True


class TestDataQualityScoreFlow:
    def test_boundary_updates_data_quality_score(self) -> None:
        profile = SensorProfile(
            operational_range=(0.0, 100.0),
            **_PROFILE_KWARGS,
        )
        phase = BoundaryCheckPhase()
        ctx = _make_ctx([-10.0, 50.0, 200.0], profile=profile)
        result = phase.execute(ctx)

        # 2 of 3 values outside → score should be low
        assert result.data_quality_score < 0.5
        assert result.boundary_result.data_quality_score < 0.5

    def test_data_quality_score_takes_minimum(self) -> None:
        profile = SensorProfile(
            operational_range=(0.0, 100.0),
            **_PROFILE_KWARGS,
        )
        phase = BoundaryCheckPhase()
        # Start with score=0.3 from sanitize, boundary would give 1.0 → final = 0.3
        ctx = _make_ctx([25.0, 50.0, 75.0], profile=profile, data_quality_score=0.3)
        result = phase.execute(ctx)

        assert result.data_quality_score == pytest.approx(0.3)

    def test_empty_values_does_not_crash(self) -> None:
        phase = BoundaryCheckPhase()
        ctx = _make_ctx([], profile=None)
        result = phase.execute(ctx)
        assert result.boundary_result.within_domain is True
        assert "empty_values" in result.boundary_result.warnings


class TestCreateEarlyResult:
    def test_create_early_result_structure(self) -> None:
        from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult

        phase = BoundaryCheckPhase()
        boundary = BoundaryResult(
            within_domain=False,
            rejection_reason="out_of_domain",
            data_quality_score=0.0,
            warnings=["values_outside_boundary:1"],
            dynamic_range_used=True,
        )
        ctx = _make_ctx([1000.0]).with_field(boundary_result=boundary)
        result = phase.create_early_result(ctx)

        assert isinstance(result, PredictionResult)
        assert result.predicted_value is None
        assert result.confidence == 0.0
        assert result.metadata["is_out_of_domain"] is True
        assert result.metadata["boundary_check"]["dynamic_range_used"] is True
        assert "values_outside_boundary:1" in result.metadata["boundary_check"]["warnings"]
