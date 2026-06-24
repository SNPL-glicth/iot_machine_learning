"""Tests for SanitizePhase (IMP-1 + extended).

Covers:
  * NaN/Inf → per-value rejection (fallback only when ALL values rejected)
  * In-window linear interpolation when adjacent valid values exist
  * History-based imputation fallback with store
  * Spike detection (>5σ from history)
  * data_quality_score computation
  * CUSUM, bounds providers, pipeline wiring
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


# -- fake Redis ------------------------------------------------------------


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
# Core scenarios
# =========================================================================


class TestCorScenarios:
    def test_inf_values_get_imputed_in_window_when_adjacent_valid(self) -> None:
        phase = SanitizePhase()
        ctx = _make_ctx([10.0, 11.0, float("inf"), 12.0])
        result = phase.execute(ctx)
        assert "value_imputed" in result.sanitization_flags
        assert not result.is_fallback
        assert len(result.sanitized_values) == 4
        # inf at index 2 → interpolated between 11.0 and 12.0 = 11.5
        assert result.sanitized_values[2] == pytest.approx(11.5)

    def test_all_nan_triggers_fallback(self) -> None:
        phase = SanitizePhase()
        ctx = _make_ctx([float("nan"), float("nan")])
        result = phase.execute(ctx)
        assert "nan_or_inf_rejected" in result.sanitization_flags
        assert result.is_fallback is True
        assert result.fallback_reason == "nan_or_inf_rejected"
        assert result.sanitized_values == []

    def test_nan_imputed_by_history_when_no_adjacent(self) -> None:
        phase = SanitizePhase()
        ctx = _make_ctx([float("nan")])
        result = phase.execute(ctx)
        # No store → no history → value_rejected, but only NaN, so fallback
        assert "value_rejected" in result.sanitization_flags
        assert "nan_or_inf_rejected" in result.sanitization_flags
        assert result.is_fallback is True

    def test_in_window_linear_interpolation(self) -> None:
        phase = SanitizePhase()
        ctx = _make_ctx([1.0, float("nan"), 3.0, 4.0])
        result = phase.execute(ctx)
        assert "value_imputed" in result.sanitization_flags
        assert len(result.sanitized_values) == 4
        # NaN at index 1 → interpolated between 1.0 and 3.0 = 2.0
        assert result.sanitized_values[1] == pytest.approx(2.0)

    def test_nan_at_start_backfilled(self) -> None:
        phase = SanitizePhase()
        ctx = _make_ctx([float("nan"), 5.0, 6.0])
        result = phase.execute(ctx)
        assert "value_imputed" in result.sanitization_flags
        assert result.sanitized_values[0] == 5.0  # backfill from next valid

    def test_nan_at_end_forward_filled(self) -> None:
        phase = SanitizePhase()
        ctx = _make_ctx([5.0, 6.0, float("nan")])
        result = phase.execute(ctx)
        assert "value_imputed" in result.sanitization_flags
        assert result.sanitized_values[2] == 6.0  # forward fill from prev

    def test_history_imputation_when_window_cannot_interpolate(self) -> None:
        redis = _FakeRedis()
        store = SeriesValuesStore(redis_client=redis)
        store.append_many("sid", [10.0, 20.0, 30.0, 40.0, 50.0])
        phase = SanitizePhase(series_values_store=store)
        # Single NaN in isolation (no neighbours in window)
        ctx = _make_ctx([float("nan")], series_id="sid")
        result = phase.execute(ctx)
        assert "value_imputed_from_history" in result.sanitization_flags or "value_imputed" in result.sanitization_flags
        # Median of [10,20,30,40,50] = 30.0
        assert result.sanitized_values[0] == pytest.approx(30.0)

    def test_spike_detected_from_history(self) -> None:
        redis = _FakeRedis()
        store = SeriesValuesStore(redis_client=redis)
        # Tight history around 10 ± 1
        store.append_many("sid", [9.0, 10.0, 11.0] * 30)
        phase = SanitizePhase(series_values_store=store)
        ctx = _make_ctx([10.0, 1000.0, 10.0], series_id="sid")
        result = phase.execute(ctx)
        assert "spike_suspected" in result.sanitization_flags

    def test_no_false_spike_on_normal_values(self) -> None:
        redis = _FakeRedis()
        store = SeriesValuesStore(redis_client=redis)
        store.append_many("sid", [9.0, 10.0, 11.0] * 10)
        phase = SanitizePhase(series_values_store=store)
        ctx = _make_ctx([10.0, 10.5, 9.8], series_id="sid")
        result = phase.execute(ctx)
        assert "spike_suspected" not in result.sanitization_flags

    def test_data_quality_score_perfect(self) -> None:
        phase = SanitizePhase()
        ctx = _make_ctx([1.0, 2.0, 3.0, 4.0, 5.0])
        result = phase.execute(ctx)
        assert result.data_quality_score == pytest.approx(1.0)

    def test_data_quality_score_reduced_on_problematic(self) -> None:
        phase = SanitizePhase()
        # 1 NaN in 4 → imputed → 1/4 = 0.25 problematic → score = 0.75
        ctx = _make_ctx([1.0, float("nan"), 3.0, 4.0])
        result = phase.execute(ctx)
        assert result.data_quality_score == pytest.approx(0.75, abs=0.01)

    def test_data_quality_score_zero_when_all_rejected(self) -> None:
        phase = SanitizePhase()
        ctx = _make_ctx([float("nan"), float("nan")])
        result = phase.execute(ctx)
        assert result.data_quality_score == pytest.approx(0.0)

    # (4) slow linear ramp → cusum_ramp_detected fires; values pass through
    def test_slow_linear_ramp_triggers_cusum(self) -> None:
        values = [10.0 + 0.5 * i + (0.01 if i % 2 else -0.01) for i in range(20)]
        phase = SanitizePhase()
        ctx = _make_ctx(values, series_id="ramp-series")
        result = phase.execute(ctx)
        assert "cusum_ramp_detected" in result.sanitization_flags
        assert result.is_fallback is False
        assert result.sanitized_values is not None
        assert len(result.sanitized_values) == len(values)

    # (3) new series, no Redis history, short window → bounds_unavailable_skipped
    def test_new_series_no_history_skips_bounds(self) -> None:
        phase = SanitizePhase(
            config=SanitizeConfig(min_window_size=10, redis_min_samples=20),
        )
        ctx = _make_ctx([10.0, 11.0, 12.0], series_id="cold-series")
        result = phase.execute(ctx)
        assert "bounds_unavailable_skipped" in result.sanitization_flags
        assert result.sanitized_values == [10.0, 11.0, 12.0]
        assert result.is_fallback is False

    # clean input → empty flags list, values pass through unchanged
    def test_clean_input_empty_flags(self) -> None:
        phase = SanitizePhase()
        values = [10.0, 10.1, 9.9, 10.05, 9.95]
        ctx = _make_ctx(values, series_id="clean-series")
        result = phase.execute(ctx)
        assert result.sanitization_flags == []
        assert result.sanitized_values == values
        assert result.values == values
        assert result.is_fallback is False
        assert result.data_quality_score == pytest.approx(1.0)


# =========================================================================
# Clamping coverage
# =========================================================================


class TestClamping:
    def test_clamp_above_upper_bound(self) -> None:
        redis = _FakeRedis()
        store = SeriesValuesStore(redis_client=redis)
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
        store.append_many("sid", [9.0, 10.0, 11.0] * 20)
        phase = SanitizePhase(
            config=SanitizeConfig(redis_min_samples=20),
            series_values_store=store,
        )
        ctx = _make_ctx([10.0, 10.0, 10.0, 100.0], series_id="sid")
        result = phase.execute(ctx)
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
        # Just verify the default phase list ordering without constructing
        assert PipelineExecutor.__init__.__defaults__ is not None
        # Alternative: check factory directly
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor_factory import (
            PipelineExecutorFactory,
        )
        factory = PipelineExecutorFactory()

        class _MockFlags:
            ML_DECISION_ARBITER_ENABLED = False
            ML_COHERENCE_CHECK_ENABLED = False
            ML_CONFIDENCE_CALIBRATION_ENABLED = False
            ML_ACTION_GUARD_ENABLED = False
            ML_EXPLAINABILITY_ENABLED = False
            ML_NARRATIVE_ENABLED = False
            ML_DOMAIN_BOUNDARY_ENABLED = False

        executor = factory.create(flags_snapshot=_MockFlags())
        names = [p.__class__.__name__ for p in executor._phases]
        assert names[0] == "SanitizePhase"
        assert names[1] == "BoundaryCheckPhase"
        assert names[2] == "PredictionReadinessGate"
        assert names[3] == "PerceivePhase"

    def test_all_nan_triggers_sanitize_fallback_result(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor_factory import (
            PipelineExecutorFactory,
        )
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration import (
            pipeline_executor as pe_mod,
        )

        class _MockFlags:
            ML_DECISION_ARBITER_ENABLED = False
            ML_COHERENCE_CHECK_ENABLED = False
            ML_CONFIDENCE_CALIBRATION_ENABLED = False
            ML_ACTION_GUARD_ENABLED = False
            ML_EXPLAINABILITY_ENABLED = False
            ML_NARRATIVE_ENABLED = False
            ML_DOMAIN_BOUNDARY_ENABLED = False

        class _MockOrchestrator:
            _budget_ms = 500.0
            _storage = None
            _last_explanation = None
            _series_values_store = None

        factory = PipelineExecutorFactory()
        executor = factory.create(flags_snapshot=_MockFlags())
        result = executor.execute(
            orchestrator=_MockOrchestrator(),
            values=[float("nan"), float("nan")],
            timestamps=None,
            series_id="test-series",
            flags=_MockFlags(),
        )

        assert result.predicted_value is None
        assert result.confidence == 0.0
        assert result.metadata["is_sanitize_fallback"] is True
        assert result.metadata["rejection_reason"] == "nan_or_inf_rejected"
        assert "nan_or_inf_rejected" in result.metadata["sanitization_flags"]


# =========================================================================
# Imputer unit tests
# =========================================================================


class TestLinearInterpolator:
    def test_importable(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.sanitize.imputer import (
            LinearInterpolator,
        )
        imp = LinearInterpolator()
        result = imp.impute(float("nan"), [1.0, 2.0, 3.0, 4.0])
        assert result == pytest.approx(2.5)

    def test_insufficient_history_raises(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.sanitize.imputer import (
            LinearInterpolator,
        )
        imp = LinearInterpolator(min_history=5)
        with pytest.raises(ValueError):
            imp.impute(float("nan"), [1.0, 2.0])

    def test_fallback_value_on_insufficient_history(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.sanitize.imputer import (
            LinearInterpolator,
        )
        imp = LinearInterpolator(min_history=5, fallback_value=42.0)
        result = imp.impute(float("nan"), [1.0, 2.0])
        assert result == 42.0
