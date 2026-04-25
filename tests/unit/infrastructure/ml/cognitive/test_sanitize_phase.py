"""Tests for SanitizePhase (IMP-1).

Covers the four scenarios mandated by the spec plus unit coverage of
the CUSUM detector, bounds providers, and pipeline wiring.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.sanitize import (
    LocalWindowBoundsProvider,
    SanitizeConfig,
    SanitizePhase,
    SeriesValuesBoundsProvider,
    detect_ramp,
)
from iot_machine_learning.infrastructure.ml.cognitive.series_values import (
    SeriesValuesStore,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import (
    create_initial_context,
)


# -- fake Redis (minimal; shared with SeriesValuesStore tests) ------------


class _FakePipeline:
    def __init__(self, store, ttls):
        self._store = store
        self._ttls = ttls
        self._ops = []

    def rpush(self, key, *values):
        self._ops.append(("rpush", key, list(values)))
        return self

    def ltrim(self, key, start, end):
        self._ops.append(("ltrim", key, start, end))
        return self

    def expire(self, key, ttl):
        self._ops.append(("expire", key, int(ttl)))
        return self

    def execute(self):
        for op in self._ops:
            if op[0] == "rpush":
                self._store.setdefault(op[1], []).extend(op[2])
            elif op[0] == "ltrim":
                _, key, start, end = op
                lst = self._store.get(key, [])
                if end == -1:
                    self._store[key] = lst[start:]
                else:
                    self._store[key] = lst[start : end + 1]
            elif op[0] == "expire":
                self._ttls[op[1]] = op[2]


class _FakeRedis:
    def __init__(self):
        self.store = {}
        self.ttls = {}

    def pipeline(self):
        return _FakePipeline(self.store, self.ttls)

    def lrange(self, key, start, end):
        lst = self.store.get(key, [])
        if end == -1:
            return list(lst[start:])
        return list(lst[start : end + 1])

    def delete(self, key):
        return 1 if self.store.pop(key, None) is not None else 0


def _make_ctx(values, series_id="s-1"):
    return create_initial_context(
        orchestrator=None,
        values=values,
        timestamps=None,
        series_id=series_id,
        flags=None,
        timer=None,
    )


# =========================================================================
# Four mandated scenarios
# =========================================================================


class TestMandatoryScenarios:
    """The four test cases explicitly required by the IMP-1 spec."""

    # (1) float('inf') → flag + fallback, no exception, pipeline does not crash
    def test_inf_in_values_flags_and_fallback(self) -> None:
        phase = SanitizePhase()
        ctx = _make_ctx([10.0, 11.0, float("inf"), 12.0])

        result = phase.execute(ctx)

        assert "nan_or_inf_rejected" in result.sanitization_flags
        assert result.is_fallback is True
        assert result.fallback_reason == "nan_or_inf_rejected"
        assert result.sanitized_values == []

    def test_nan_in_values_flags_and_fallback(self) -> None:
        phase = SanitizePhase()
        ctx = _make_ctx([10.0, float("nan"), 12.0])

        result = phase.execute(ctx)

        assert "nan_or_inf_rejected" in result.sanitization_flags
        assert result.is_fallback is True

    # (2) slow linear ramp → cusum_ramp_detected fires; values pass through
    def test_slow_linear_ramp_triggers_cusum(self) -> None:
        # 20 points, clear drift. Added small noise so σ > 0 without killing the trend.
        values = [10.0 + 0.5 * i + (0.01 if i % 2 else -0.01) for i in range(20)]
        phase = SanitizePhase()
        ctx = _make_ctx(values, series_id="ramp-series")

        result = phase.execute(ctx)

        assert "cusum_ramp_detected" in result.sanitization_flags
        assert result.is_fallback is False
        # CUSUM should NOT block — values flow through.
        assert result.sanitized_values is not None
        assert len(result.sanitized_values) == len(values)

    # (3) new series, no Redis history, short window → bounds_unavailable_skipped
    def test_new_series_no_history_skips_bounds(self) -> None:
        phase = SanitizePhase(
            config=SanitizeConfig(min_window_size=10, redis_min_samples=20),
        )
        ctx = _make_ctx([10.0, 11.0, 12.0], series_id="cold-series")  # < 10

        result = phase.execute(ctx)

        assert "bounds_unavailable_skipped" in result.sanitization_flags
        assert result.sanitized_values == [10.0, 11.0, 12.0]
        assert result.is_fallback is False

    # (4) clean input → empty flags list, values pass through unchanged
    def test_clean_input_empty_flags(self) -> None:
        phase = SanitizePhase()
        values = [10.0, 10.1, 9.9, 10.05, 9.95]
        ctx = _make_ctx(values, series_id="clean-series")

        result = phase.execute(ctx)

        assert result.sanitization_flags == []
        assert result.sanitized_values == values
        assert result.values == values
        assert result.is_fallback is False


# =========================================================================
# Clamping coverage
# =========================================================================


class TestClamping:
    def test_clamp_above_upper_bound(self) -> None:
        redis = _FakeRedis()
        store = SeriesValuesStore(redis_client=redis)
        # Seed tight history around 10 (σ≈0.82) → 6σ bound ≈ 15
        seed = [9.0, 10.0, 11.0] * 20
        store.append_many("s", seed)
        phase = SanitizePhase(
            config=SanitizeConfig(redis_min_samples=20),
            series_values_store=store,
        )
        ctx = _make_ctx([10.0, 10.0, 10.0, 1000.0], series_id="s")
        result = phase.execute(ctx)
        assert result.sanitized_values[-1] < 1000.0
        assert any(f.startswith("value_clamped:") for f in result.sanitization_flags)

    def test_explicit_bounds_from_store(self) -> None:
        redis = _FakeRedis()
        store = SeriesValuesStore(redis_client=redis)
        # Tight history around 10 with σ≈1 → 6σ bounds ≈ [4, 16]
        store.append_many("sid", [9.0, 10.0, 11.0] * 20)
        phase = SanitizePhase(
            config=SanitizeConfig(redis_min_samples=20),
            series_values_store=store,
        )
        ctx = _make_ctx([10.0, 10.0, 10.0, 100.0], series_id="sid")
        result = phase.execute(ctx)
        # 100 clamped to the upper Redis-derived bound.
        assert result.sanitized_values[-1] < 100.0
        assert any(f.startswith("value_clamped:") for f in result.sanitization_flags)


# =========================================================================
# Exception safety
# =========================================================================


class _BrokenProvider:
    def get_bounds(self, *_a, **_k):
        raise RuntimeError("provider blew up")


class TestExceptionSafety:
    def test_exception_from_provider_is_swallowed(self) -> None:
        phase = SanitizePhase(
            primary_provider=_BrokenProvider(),
            fallback_provider=_BrokenProvider(),
        )
        ctx = _make_ctx([10.0, 11.0, 12.0])
        result = phase.execute(ctx)

        # Never raises; either runs via swallowed-exception path or continues.
        assert "sanitize_exception_swallowed" in result.sanitization_flags

    def test_empty_input_no_crash(self) -> None:
        phase = SanitizePhase()
        ctx = _make_ctx([])
        result = phase.execute(ctx)
        assert result.sanitization_flags == []
        assert result.sanitized_values is None


# =========================================================================
# CUSUM unit tests
# =========================================================================


class TestCUSUM:
    def test_flat_signal_no_ramp(self) -> None:
        assert detect_ramp([10.0] * 20) is False

    def test_tiny_window_no_ramp(self) -> None:
        assert detect_ramp([10.0, 11.0, 12.0]) is False

    def test_pure_ramp_detected(self) -> None:
        assert detect_ramp([float(i) for i in range(30)]) is True

    def test_random_noise_does_not_ramp(self) -> None:
        import random

        random.seed(0)
        values = [10.0 + random.gauss(0, 0.01) for _ in range(50)]
        assert detect_ramp(values) is False


# =========================================================================
# Bounds providers
# =========================================================================


class TestLocalWindowBoundsProvider:
    def test_insufficient_window_returns_none(self) -> None:
        p = LocalWindowBoundsProvider(min_window_size=10)
        assert p.get_bounds("s", [1.0, 2.0], 6.0) is None

    def test_zero_variance_returns_none(self) -> None:
        p = LocalWindowBoundsProvider(min_window_size=3)
        assert p.get_bounds("s", [5.0, 5.0, 5.0, 5.0], 6.0) is None

    def test_happy_path_returns_bounds(self) -> None:
        p = LocalWindowBoundsProvider(min_window_size=3)
        bounds = p.get_bounds("s", [1.0, 2.0, 3.0, 4.0, 5.0], 6.0)
        assert bounds is not None
        lower, upper = bounds
        assert lower < 3.0 < upper


class TestSeriesValuesBoundsProvider:
    def test_no_history_returns_none(self) -> None:
        redis = _FakeRedis()
        store = SeriesValuesStore(redis_client=redis)
        p = SeriesValuesBoundsProvider(store, min_samples=20)
        assert p.get_bounds("new", [1.0, 2.0, 3.0], 6.0) is None

    def test_history_returns_bounds(self) -> None:
        redis = _FakeRedis()
        store = SeriesValuesStore(redis_client=redis)
        store.append_many("sid", [10.0 + (i % 5) * 0.1 for i in range(30)])
        p = SeriesValuesBoundsProvider(store, min_samples=20)
        bounds = p.get_bounds("sid", [10.0], 6.0)
        assert bounds is not None


# =========================================================================
# Pipeline wiring — SanitizePhase must be phase [0]
# =========================================================================


class TestPipelineWiring:
    def test_sanitize_registered_as_phase_zero(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor import (
            PipelineExecutor,
        )

        executor = PipelineExecutor()
        assert executor._phases[0].__class__.__name__ == "SanitizePhase"
        assert executor._phases[1].__class__.__name__ == "BoundaryCheckPhase"
        assert executor._phases[2].__class__.__name__ == "PerceivePhase"

    def test_inf_triggers_sanitize_fallback_result(self) -> None:
        """End-to-end: float('inf') → PredictionResult with is_sanitize_fallback."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor import (
            PipelineExecutor,
        )

        class _MockOrchestrator:
            _budget_ms = 500.0
            _storage = None
            _last_explanation = None
            _series_values_store = None

        class _MockFlags:
            ML_DOMAIN_BOUNDARY_ENABLED = False

        executor = PipelineExecutor()
        result = executor.execute(
            orchestrator=_MockOrchestrator(),
            values=[1.0, 2.0, float("inf"), 3.0],
            timestamps=None,
            series_id="test-series",
            flags=_MockFlags(),
        )

        # Sanitize-fallback short-circuits the pipeline.
        assert result.predicted_value is None
        assert result.confidence == 0.0
        assert result.metadata["is_sanitize_fallback"] is True
        assert result.metadata["rejection_reason"] == "nan_or_inf_rejected"
        assert "nan_or_inf_rejected" in result.metadata["sanitization_flags"]
