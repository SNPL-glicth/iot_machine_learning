"""Tests for pipeline latency instrumentation and architectural guards.

Covers:
    ⚠️1 — PipelineTimer per-phase timing + budget guard
    ⚠️2 — Orchestrator complexity guard (line count)
    ⚠️3 — Legacy sunset: deprecation warnings on legacy APIs
"""

from __future__ import annotations

import inspect
import time
import warnings
from unittest.mock import MagicMock

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import PipelineTimer


# ── ⚠️1: PipelineTimer unit tests ─────────────────────────────


class TestPipelineTimerBasics:
    """PipelineTimer records per-phase latency and checks budget."""

    def test_initial_state_all_zeros(self) -> None:
        t = PipelineTimer()
        assert t.total_ms == 0.0
        assert not t.is_over_budget

    def test_default_budget(self) -> None:
        t = PipelineTimer()
        assert t.budget_ms == 500.0

    def test_custom_budget(self) -> None:
        t = PipelineTimer(budget_ms=100.0)
        assert t.budget_ms == 100.0

    def test_start_stop_records_phase(self) -> None:
        t = PipelineTimer()
        t.start()
        time.sleep(0.005)  # ~5ms
        elapsed = t.stop("perceive")
        assert elapsed > 0.0
        assert t.perceive_ms == elapsed
        assert t.total_ms > 0.0

    def test_multiple_phases_accumulate(self) -> None:
        t = PipelineTimer()
        for phase in ("perceive", "predict", "inhibit", "adapt", "fuse", "explain"):
            t.start()
            t.stop(phase)
        assert t.total_ms >= 0.0
        assert t.perceive_ms >= 0.0
        assert t.predict_ms >= 0.0

    def test_over_budget_detection(self) -> None:
        t = PipelineTimer(budget_ms=0.001)  # 0.001ms = effectively instant
        t.start()
        time.sleep(0.002)
        t.stop("perceive")
        assert t.is_over_budget

    def test_under_budget(self) -> None:
        t = PipelineTimer(budget_ms=10000.0)
        t.start()
        t.stop("perceive")
        assert not t.is_over_budget

    def test_unknown_phase_ignored(self) -> None:
        t = PipelineTimer()
        t.start()
        elapsed = t.stop("nonexistent_phase")
        assert elapsed > 0.0
        assert t.total_ms == 0.0  # nothing recorded

    def test_to_dict_keys(self) -> None:
        t = PipelineTimer()
        d = t.to_dict()
        expected_keys = {
            "perceive_ms", "predict_ms", "inhibit_ms", "adapt_ms",
            "fuse_ms", "explain_ms", "total_ms", "budget_ms", "over_budget",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_rounded(self) -> None:
        t = PipelineTimer()
        t.start()
        t.stop("perceive")
        d = t.to_dict()
        # All numeric values should be rounded to 3 decimals
        for key in ("perceive_ms", "total_ms"):
            val = d[key]
            assert isinstance(val, float)
            # Check it's rounded (no more than 3 decimal places)
            assert val == round(val, 3)


# ── ⚠️1: Orchestrator timing integration ──────────────────────


class TestOrchestratorPipelineTiming:
    """Orchestrator exposes per-phase timing in metadata."""

    def _make_engine(self, name: str, value: float) -> MagicMock:
        from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult
        eng = MagicMock()
        eng.name = name
        eng.can_handle.return_value = True
        eng.predict.return_value = PredictionResult(
            predicted_value=value, confidence=0.8, trend="stable",
            metadata={"diagnostic": {}},
        )
        return eng

    def test_predict_includes_pipeline_timing(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration import (
            MetaCognitiveOrchestrator,
        )
        eng = self._make_engine("test_eng", 25.0)
        orch = MetaCognitiveOrchestrator(engines=[eng], enable_plasticity=False)
        result = orch.predict([20.0, 21.0, 22.0, 23.0, 24.0])
        assert "pipeline_timing" in result.metadata
        timing = result.metadata["pipeline_timing"]
        assert "total_ms" in timing
        assert "perceive_ms" in timing
        assert "predict_ms" in timing
        assert timing["total_ms"] >= 0.0

    def test_last_pipeline_timing_property(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration import (
            MetaCognitiveOrchestrator,
        )
        eng = self._make_engine("test_eng", 25.0)
        orch = MetaCognitiveOrchestrator(engines=[eng], enable_plasticity=False)
        assert orch.last_pipeline_timing is None
        orch.predict([20.0, 21.0, 22.0, 23.0, 24.0])
        timer = orch.last_pipeline_timing
        assert timer is not None
        assert timer.total_ms > 0.0

    def test_fallback_includes_timing(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration import (
            MetaCognitiveOrchestrator,
        )
        eng = MagicMock()
        eng.name = "failing"
        eng.can_handle.return_value = True
        eng.predict.side_effect = RuntimeError("boom")
        orch = MetaCognitiveOrchestrator(engines=[eng], enable_plasticity=False)
        result = orch.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        assert "pipeline_timing" in result.metadata

    def test_budget_ms_configurable(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration import (
            MetaCognitiveOrchestrator,
        )
        eng = self._make_engine("test_eng", 25.0)
        orch = MetaCognitiveOrchestrator(
            engines=[eng], enable_plasticity=False, budget_ms=999.0,
        )
        orch.predict([20.0, 21.0, 22.0, 23.0, 24.0])
        assert orch.last_pipeline_timing.budget_ms == 999.0


# ── ⚠️2: Orchestrator complexity guard ────────────────────────


class TestOrchestratorComplexityGuard:
    """Meta-test: orchestrator.py must stay lean.

    The orchestrator DELEGATES, it never IMPLEMENTS.
    If it grows beyond the limit, it's absorbing responsibilities
    that belong in sub-modules.
    
    After refactoring (FASE 7 + Thread Safety):
    - Extracted AdvancedPlasticityCoordinator (~180 lines)
    - Extracted WeightAdjustmentService (~120 lines)
    - Extracted orchestrator_helpers (~100 lines)
    - Current: 316 lines (was 557)
    """

    _MAX_LINES = 330  # Refactored from 557 to 330 (extracted 3 modules: ~230 lines)

    def test_orchestrator_line_count(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive import orchestrator
        source = inspect.getsource(orchestrator)
        line_count = len(source.splitlines())
        assert line_count <= self._MAX_LINES, (
            f"orchestrator.py has {line_count} lines (limit: {self._MAX_LINES}). "
            f"The orchestrator delegates, it never implements. "
            f"Extract new logic into sub-modules."
        )

    def test_orchestrator_no_direct_math(self) -> None:
        """Orchestrator should not contain numpy/scipy/math-heavy code."""
        from iot_machine_learning.infrastructure.ml.cognitive import orchestrator
        source = inspect.getsource(orchestrator)
        forbidden = ["import numpy", "import scipy", "from numpy", "from scipy"]
        for pattern in forbidden:
            assert pattern not in source, (
                f"orchestrator.py imports {pattern}. "
                f"Math belongs in sub-modules, not the orchestrator."
            )

    def test_orchestrator_delegates_to_submodules(self) -> None:
        """Verify the orchestrator uses its sub-modules."""
        from iot_machine_learning.infrastructure.ml.cognitive import orchestrator
        source = inspect.getsource(orchestrator)
        required_delegates = [
            "SignalAnalyzer", "InhibitionGate", "WeightedFusion",
            "ExplanationBuilder", "PlasticityTracker",
        ]
        for delegate in required_delegates:
            assert delegate in source, (
                f"orchestrator.py does not reference {delegate}. "
                f"Has delegation been broken?"
            )


# ── ⚠️3: Legacy deprecation warnings ─────────────────────────


class TestLegacyDeprecationWarnings:
    """Verify that legacy APIs emit DeprecationWarning."""

    def test_last_diagnostic_warns(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration import (
            MetaCognitiveOrchestrator,
        )
        eng = MagicMock()
        eng.name = "e"
        eng.can_handle.return_value = False
        orch = MetaCognitiveOrchestrator(engines=[eng], enable_plasticity=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = orch.last_diagnostic
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_select_engine_for_sensor_warns(self) -> None:
        from iot_machine_learning.application.use_cases.select_engine import (
            select_engine_for_sensor,
        )
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            select_engine_for_sensor(sensor_id=1, flags=flags)
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_get_default_range_warns(self) -> None:
        from iot_machine_learning.domain.entities.iot.sensor_ranges import (
            get_default_range,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_default_range("temperature")
            assert any(issubclass(x.category, DeprecationWarning) for x in w)


# ── ⚠️3: Migration scorecard — legacy vs agnostic call-sites ──


class TestMigrationScorecard:
    """Count legacy vs agnostic patterns in the codebase.

    This is an informational meta-test. It does NOT fail — it reports
    the current migration state so developers can track progress.
    """

    def test_report_legacy_vs_agnostic_usage(self) -> None:
        """Scan domain/ports for legacy vs agnostic method usage."""
        import re
        from iot_machine_learning.domain.ports import (
            anomaly_detection_port,
            pattern_detection_port,
            prediction_port,
        )

        legacy_pattern = re.compile(r'def (predict|detect|detect_pattern)\(')
        agnostic_pattern = re.compile(
            r'def (predict_series|detect_series|detect_pattern_series)\('
        )

        legacy_count = 0
        agnostic_count = 0

        for mod in [prediction_port, anomaly_detection_port, pattern_detection_port]:
            source = inspect.getsource(mod)
            legacy_count += len(legacy_pattern.findall(source))
            agnostic_count += len(agnostic_pattern.findall(source))

        # Both should exist (dual interface)
        assert legacy_count > 0, "Legacy methods missing — dual interface broken"
        assert agnostic_count > 0, "Agnostic methods missing — migration incomplete"
        # Informational: print ratio
        total = legacy_count + agnostic_count
        agnostic_pct = (agnostic_count / total * 100) if total else 0
        # This test always passes — it's a scorecard
        assert True, (
            f"Migration scorecard: {agnostic_count}/{total} "
            f"({agnostic_pct:.0f}%) agnostic methods"
        )
