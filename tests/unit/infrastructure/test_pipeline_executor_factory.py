"""Tests for PipelineExecutorFactory (IMP-3).

Covers:
* ``create()`` returns a fresh :class:`PipelineExecutor` per call.
* Two concurrent calls get independent ``_phases`` lists — no shared
  mutable state across invocations.
* ``flags_snapshot`` is accepted but does not yet affect phase
  composition (documented invariant; test enforces current behaviour
  so future flag-driven changes force an update here).
* The module-level ``_pipeline_executor`` singleton no longer exists.
* ``execute_pipeline`` prefers the orchestrator's factory and
  constructs a one-shot factory when none is present.
"""

from __future__ import annotations

import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List
from unittest.mock import MagicMock

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor import (
    PipelineExecutor,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor_factory import (
    PipelineExecutorFactory,
)


# =========================================================================
# Factory behaviour
# =========================================================================


class TestPipelineExecutorFactory:
    def test_create_returns_pipeline_executor(self) -> None:
        factory = PipelineExecutorFactory()
        executor = factory.create()
        assert isinstance(executor, PipelineExecutor)

    def test_create_returns_fresh_instance_each_call(self) -> None:
        factory = PipelineExecutorFactory()
        a = factory.create()
        b = factory.create()
        assert a is not b
        # Fresh phase lists too — not shared via class attribute.
        assert a._phases is not b._phases
        assert a._assembly is not b._assembly

    def test_phases_are_independent_instances(self) -> None:
        """Mutating one executor's phase list must not leak to the next."""
        factory = PipelineExecutorFactory()
        a = factory.create()
        b = factory.create()
        a._phases.clear()
        assert len(b._phases) > 0

    def test_flags_snapshot_accepted_but_phase_list_static(self) -> None:
        """Today flags_snapshot does not influence phase composition.
        
        When future flag-driven phase selection is introduced, this test
        must be updated to assert the expected variation. Leaving it
        green acts as a tripwire.
        """
        factory = PipelineExecutorFactory()
        flags_a = MagicMock()
        flags_a.ML_DOMAIN_BOUNDARY_ENABLED = True
        flags_b = MagicMock()
        flags_b.ML_DOMAIN_BOUNDARY_ENABLED = False

        exec_a = factory.create(flags_a)
        exec_b = factory.create(flags_b)

        types_a = [type(p).__name__ for p in exec_a._phases]
        types_b = [type(p).__name__ for p in exec_b._phases]
        assert types_a == types_b, (
            "Phase composition unexpectedly diverged — update "
            "PipelineExecutorFactory.create to document the new behaviour."
        )

    def test_phase_order_matches_executor_contract(self) -> None:
        factory = PipelineExecutorFactory()
        executor = factory.create()
        names = [type(p).__name__ for p in executor._phases]
        # Phase [0] must be SanitizePhase (IMP-1 invariant).
        assert names[0] == "SanitizePhase"
        assert names[1] == "BoundaryCheckPhase"
        assert names[2] == "PerceivePhase"


# =========================================================================
# Singleton removal + concurrency
# =========================================================================


class TestSingletonRemoval:
    def test_module_level_singleton_removed(self) -> None:
        """_pipeline_executor must no longer exist at module scope."""
        import iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor as pe_mod

        assert not hasattr(pe_mod, "_pipeline_executor"), (
            "IMP-3 removed the module-level _pipeline_executor singleton. "
            "Use PipelineExecutorFactory.create() instead."
        )

    def test_concurrent_creates_produce_distinct_executors(self) -> None:
        """Two threads calling create() concurrently get distinct objects."""
        factory = PipelineExecutorFactory()
        barrier = threading.Barrier(8)
        instances: List[PipelineExecutor] = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            inst = factory.create()
            with lock:
                instances.append(inst)

        with ThreadPoolExecutor(max_workers=8) as pool:
            for _ in range(8):
                pool.submit(worker)

        assert len(instances) == 8
        # All distinct instances.
        assert len({id(i) for i in instances}) == 8
        # All distinct phase lists.
        assert len({id(i._phases) for i in instances}) == 8


# =========================================================================
# execute_pipeline integration
# =========================================================================


class TestExecutePipelineIntegration:
    def test_execute_pipeline_uses_orchestrator_factory(self) -> None:
        """execute_pipeline must prefer orchestrator._pipeline_executor_factory."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration import (
            pipeline_executor as pe_mod,
        )

        created: List[PipelineExecutor] = []

        class SpyFactory(PipelineExecutorFactory):
            def create(self, flags_snapshot=None):  # type: ignore[override]
                exec_ = super().create(flags_snapshot)
                created.append(exec_)
                return exec_

        class _MockFlags:
            ML_DOMAIN_BOUNDARY_ENABLED = False

        class _MockOrchestrator:
            _budget_ms = 500.0
            _storage = None
            _last_explanation = None
            _series_values_store = None

            def __init__(self):
                self._pipeline_executor_factory = SpyFactory()

        orch = _MockOrchestrator()
        # Trigger a sanitize-fallback short-circuit so we do not have to
        # wire up every phase's dependencies — NaN input exits the
        # pipeline on phase [0].
        result = pe_mod.execute_pipeline(
            orch,
            values=[1.0, 2.0, float("nan")],
            timestamps=None,
            series_id="test",
            flags_snapshot=_MockFlags(),
        )

        assert result.metadata.get("is_sanitize_fallback") is True
        assert len(created) == 1  # exactly one executor instantiated

    def test_execute_pipeline_works_without_factory_on_orchestrator(self) -> None:
        """Fallback path constructs a one-shot factory when orchestrator
        has no _pipeline_executor_factory attribute (back-compat)."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration import (
            pipeline_executor as pe_mod,
        )

        class _MockFlags:
            ML_DOMAIN_BOUNDARY_ENABLED = False

        class _BareOrchestrator:
            _budget_ms = 500.0
            _storage = None
            _last_explanation = None
            _series_values_store = None
            # deliberately no _pipeline_executor_factory

        result = pe_mod.execute_pipeline(
            _BareOrchestrator(),
            values=[1.0, 2.0, float("inf")],
            timestamps=None,
            series_id="test",
            flags_snapshot=_MockFlags(),
        )
        assert result.metadata.get("is_sanitize_fallback") is True


# =========================================================================
# Orchestrator wiring
# =========================================================================


class TestOrchestratorWiring:
    def test_orchestrator_owns_a_factory(self) -> None:
        """MetaCognitiveOrchestrator must expose _pipeline_executor_factory."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import (
            MetaCognitiveOrchestrator,
        )

        # Minimal mock engine to satisfy "at least one engine required".
        engine = MagicMock()
        engine.name = "mock"
        engine.can_handle = lambda n: True

        orch = MetaCognitiveOrchestrator(engines=[engine])
        assert hasattr(orch, "_pipeline_executor_factory")
        assert isinstance(orch._pipeline_executor_factory, PipelineExecutorFactory)
