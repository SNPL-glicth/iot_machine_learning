"""Tests for SanitizePhase (IMP-1).

Covers:
- Happy path: values pass through untouched
- NaN/Inf hard-stop rejection
- Out-of-range clamping using per-series history
- New series without history (local window fallback)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.sanitize.phase import (
    LocalWindowStatisticsProvider,
    SanitizeConfig,
    SanitizePhase,
    SeriesStatisticsProvider,
)


class FakeSeriesStatisticsProvider(SeriesStatisticsProvider):
    """Provider that returns pre-configured statistics per series."""

    def __init__(self, stats_map: dict) -> None:
        self._stats_map = stats_map

    def get_statistics(self, series_id: str) -> Optional[Tuple[float, float]]:
        return self._stats_map.get(series_id)


class TestSanitizePhaseHappyPath:
    """Values within bounds pass through unchanged."""

    def test_clean_values_no_clamp(self) -> None:
        phase = SanitizePhase()
        values = [10.0, 11.0, 12.0, 10.5, 11.5]

        sanitized, flags = phase._sanitize_values(values, "s-1", [])

        assert sanitized == values
        assert flags == []

    def test_empty_input(self) -> None:
        phase = SanitizePhase()
        sanitized, flags = phase._sanitize_values([], "s-1", [])

        assert sanitized == []
        assert flags == []


class TestSanitizePhaseNaNInfRejection:
    """Hard-stop on non-finite values."""

    def test_nan_rejected(self) -> None:
        phase = SanitizePhase()
        values = [1.0, float("nan"), 3.0]

        sanitized, flags = phase._sanitize_values(values, "s-1", [])

        assert "nan_or_inf_rejected" not in flags  # _sanitize_values doesn't check
        # The execute() method does the NaN/Inf check before _sanitize_values
        # so we test via execute() instead

    def test_execute_nan_sets_fallback(self) -> None:
        from dataclasses import dataclass, field
        from typing import Any, Dict, List, Optional

        @dataclass
        class FakeCtx:
            orchestrator: Any = None
            values: List[float] = field(default_factory=list)
            timestamps: Optional[List[float]] = None
            series_id: str = ""
            flags: Any = None
            timer: Any = None
            domain: Optional[str] = None
            metadata: Dict[str, Any] = field(default_factory=dict)
            sanitized_values: Optional[List[float]] = None
            sanitization_flags: List[str] = field(default_factory=list)
            degradation_reasons: List[str] = field(default_factory=list)
            is_fallback: bool = False
            fallback_reason: Optional[str] = None

            def with_field(self, **kwargs) -> "FakeCtx":
                current = {k: getattr(self, k) for k in self.__dataclass_fields__}
                current.update(kwargs)
                return FakeCtx(**current)

        phase = SanitizePhase()
        ctx = FakeCtx(values=[1.0, float("nan"), 3.0], series_id="s-1")
        result = phase.execute(ctx)

        assert result.is_fallback is True
        assert result.fallback_reason == "nan_or_inf_rejected"
        assert "nan_or_inf_rejected" in result.sanitization_flags
        assert result.sanitized_values == []

    def test_execute_inf_sets_fallback(self) -> None:
        from dataclasses import dataclass, field
        from typing import Any, Dict, List, Optional

        @dataclass
        class FakeCtx:
            orchestrator: Any = None
            values: List[float] = field(default_factory=list)
            timestamps: Optional[List[float]] = None
            series_id: str = ""
            flags: Any = None
            timer: Any = None
            domain: Optional[str] = None
            metadata: Dict[str, Any] = field(default_factory=dict)
            sanitized_values: Optional[List[float]] = None
            sanitization_flags: List[str] = field(default_factory=list)
            degradation_reasons: List[str] = field(default_factory=list)
            is_fallback: bool = False
            fallback_reason: Optional[str] = None

            def with_field(self, **kwargs) -> "FakeCtx":
                current = {k: getattr(self, k) for k in self.__dataclass_fields__}
                current.update(kwargs)
                return FakeCtx(**current)

        phase = SanitizePhase()
        ctx = FakeCtx(values=[1.0, float("inf"), 3.0], series_id="s-1")
        result = phase.execute(ctx)

        assert result.is_fallback is True
        assert result.fallback_reason == "nan_or_inf_rejected"
        assert "nan_or_inf_rejected" in result.sanitization_flags

    def test_execute_neg_inf_sets_fallback(self) -> None:
        from dataclasses import dataclass, field
        from typing import Any, Dict, List, Optional

        @dataclass
        class FakeCtx:
            orchestrator: Any = None
            values: List[float] = field(default_factory=list)
            timestamps: Optional[List[float]] = None
            series_id: str = ""
            flags: Any = None
            timer: Any = None
            domain: Optional[str] = None
            metadata: Dict[str, Any] = field(default_factory=dict)
            sanitized_values: Optional[List[float]] = None
            sanitization_flags: List[str] = field(default_factory=list)
            degradation_reasons: List[str] = field(default_factory=list)
            is_fallback: bool = False
            fallback_reason: Optional[str] = None

            def with_field(self, **kwargs) -> "FakeCtx":
                current = {k: getattr(self, k) for k in self.__dataclass_fields__}
                current.update(kwargs)
                return FakeCtx(**current)

        phase = SanitizePhase()
        ctx = FakeCtx(values=[1.0, float("-inf"), 3.0], series_id="s-1")
        result = phase.execute(ctx)

        assert result.is_fallback is True
        assert result.fallback_reason == "nan_or_inf_rejected"


class TestSanitizePhaseClamping:
    """Values outside [mean - 6σ, mean + 6σ] are clamped."""

    def test_clamp_above_upper_bound(self) -> None:
        provider = FakeSeriesStatisticsProvider({"s-1": (10.0, 1.0)})
        phase = SanitizePhase(statistics_provider=provider)

        values = [10.0, 10.0, 10.0, 17.0]  # 17 > 10 + 6*1 = 16
        sanitized, flags = phase._sanitize_values(values, "s-1", [])

        assert sanitized[-1] == 16.0  # clamped to upper bound
        assert sanitized[:3] == [10.0, 10.0, 10.0]
        assert "value_clamped:1" in flags

    def test_clamp_below_lower_bound(self) -> None:
        provider = FakeSeriesStatisticsProvider({"s-1": (10.0, 1.0)})
        phase = SanitizePhase(statistics_provider=provider)

        values = [3.0, 10.0, 10.0, 10.0]  # 3 < 10 - 6*1 = 4
        sanitized, flags = phase._sanitize_values(values, "s-1", [])

        assert sanitized[0] == 4.0  # clamped to lower bound
        assert "value_clamped:1" in flags

    def test_multiple_clamped(self) -> None:
        provider = FakeSeriesStatisticsProvider({"s-1": (0.0, 1.0)})
        phase = SanitizePhase(statistics_provider=provider)

        values = [-10.0, 0.0, 10.0]  # both -10 and 10 outside [-6, 6]
        sanitized, flags = phase._sanitize_values(values, "s-1", [])

        assert sanitized[0] == -6.0
        assert sanitized[1] == 0.0
        assert sanitized[2] == 6.0
        assert "value_clamped:2" in flags


class TestSanitizePhaseNoHistory:
    """Series without Redis history falls back to local window statistics."""

    def test_local_window_computes_std(self) -> None:
        phase = SanitizePhase()  # default LocalWindowStatisticsProvider
        values = [10.0, 10.0, 10.0, 10.0, 100.0]  # last value is outlier

        sanitized, flags = phase._sanitize_values(values, "new-series", [])

        mean = 28.0
        variance = sum((v - mean) ** 2 for v in values) / 5  # ~1296
        std = math.sqrt(variance)  # ~36.0
        upper = mean + 6 * std  # ~244

        assert sanitized == values  # 100 < 244, no clamp
        assert flags == []

    def test_local_window_clamps_outlier(self) -> None:
        phase = SanitizePhase()
        # 40 values of 10.0 + one outlier at 500.0
        # mean = 900/41 ≈ 21.95, std ≈ 75.6, upper ≈ 475.5
        # 500 > 475, so it gets clamped
        values = [10.0] * 40 + [500.0]

        sanitized, flags = phase._sanitize_values(values, "new-series", [])

        assert sanitized[-1] < 500.0  # outlier was clamped
        assert "value_clamped:1" in flags

    def test_insufficient_window_size_no_clamp(self) -> None:
        phase = SanitizePhase(config=SanitizeConfig(min_window_size=5))
        values = [10.0, 300.0]  # only 2 values, less than min_window_size=5

        sanitized, flags = phase._sanitize_values(values, "new-series", [])

        assert sanitized == values  # no clamping, not enough data
        assert flags == []


class TestSanitizePhaseConfigurability:
    """Thresholds are configurable via constructor."""

    def test_custom_sigma_multiplier(self) -> None:
        provider = FakeSeriesStatisticsProvider({"s-1": (10.0, 1.0)})
        phase = SanitizePhase(
            config=SanitizeConfig(sigma_multiplier=3.0),
            statistics_provider=provider,
        )

        values = [14.0]  # 14 > 10 + 3*1 = 13, but < 16
        sanitized, flags = phase._sanitize_values(values, "s-1", [])

        assert sanitized[0] == 13.0  # clamped with 3σ
        assert "value_clamped:1" in flags

    def test_zero_std_passes_through(self) -> None:
        provider = FakeSeriesStatisticsProvider({"s-1": (10.0, 0.0)})
        phase = SanitizePhase(statistics_provider=provider)

        values = [10.0, 10.0, 10.0]
        sanitized, flags = phase._sanitize_values(values, "s-1", [])

        assert sanitized == values
        assert flags == []


class TestSanitizePhaseWithRealPipelineContext:
    """Integration test with actual PipelineContext from orchestration package."""

    def test_execute_modifies_real_context(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import (
            PipelineContext,
            create_initial_context,
        )

        phase = SanitizePhase()
        ctx = create_initial_context(
            orchestrator=None,
            values=[10.0, 10.0, 10.0, 100.0],
            timestamps=None,
            series_id="s-test",
            flags=None,
            timer=None,
        )

        result = phase.execute(ctx)

        assert result.sanitized_values is not None
        assert result.sanitization_flags == []  # 100 may not exceed 6σ
        assert result.values == result.sanitized_values  # downstream sees sanitized
        assert result.is_fallback is False
