"""Engine integration tests for HyperparameterAdaptor (IMP-4c).

Covers Taylor and Statistical engines only. Seasonal is intentionally
deferred to a later PR per the IMP-4c spec.

Each engine must:
1. Pick up the adaptor-loaded hyperparameters on the next ``predict()``.
2. Fall back to constructor defaults when the adaptor is inert or empty.
3. Log a ``hyperparams_loaded`` DEBUG line when params are applied.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.hyperparameters import (
    HyperparameterAdaptor,
)
from iot_machine_learning.infrastructure.ml.engines.statistical import (
    StatisticalPredictionEngine,
)
from iot_machine_learning.infrastructure.ml.engines.taylor import (
    TaylorPredictionEngine,
)


# -- minimal fake Redis ---------------------------------------------------


class _FakePipeline:
    def __init__(self, store: Dict[str, Dict[str, str]], ttls: Dict[str, int]) -> None:
        self._store = store
        self._ttls = ttls
        self._ops: List[tuple] = []

    def hset(self, key: str, mapping: Dict[str, Any]) -> "_FakePipeline":
        self._ops.append(("hset", key, dict(mapping)))
        return self

    def expire(self, key: str, ttl: int) -> "_FakePipeline":
        self._ops.append(("expire", key, int(ttl)))
        return self

    def execute(self) -> None:
        for op in self._ops:
            if op[0] == "hset":
                self._store[op[1]] = {k: str(v) for k, v in op[2].items()}
            elif op[0] == "expire":
                self._ttls[op[1]] = op[2]


class _FakeRedis:
    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, str]] = {}
        self.ttls: Dict[str, int] = {}

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self.store, self.ttls)

    def hgetall(self, key: str) -> Dict[str, str]:
        return dict(self.store.get(key, {}))

    def delete(self, key: str) -> int:
        return 1 if self.store.pop(key, None) is not None else 0


# -- Taylor ---------------------------------------------------------------


class TestTaylorHyperparams:
    def test_order_loaded_from_adaptor(self) -> None:
        adaptor = HyperparameterAdaptor(redis_client=_FakeRedis())
        adaptor.save("s1", "taylor_finite_differences", {"order": 3.0})
        engine = TaylorPredictionEngine(
            order=1,
            horizon=1,
            series_id="s1",
            hyperparameter_adaptor=adaptor,
        )
        engine.predict([20.0 + i * 0.5 for i in range(20)])
        assert engine._order == 3

    def test_defaults_when_adaptor_inert(self) -> None:
        adaptor = HyperparameterAdaptor(redis_client=None)  # inert
        engine = TaylorPredictionEngine(
            order=2,
            horizon=1,
            series_id="s1",
            hyperparameter_adaptor=adaptor,
        )
        engine.predict([20.0 + i * 0.5 for i in range(20)])
        assert engine._order == 2

    def test_defaults_when_no_series_id(self) -> None:
        adaptor = HyperparameterAdaptor(redis_client=_FakeRedis())
        adaptor.save("s1", "taylor_finite_differences", {"order": 3.0})
        engine = TaylorPredictionEngine(
            order=2,
            horizon=1,
            hyperparameter_adaptor=adaptor,
        )
        engine.predict([20.0 + i * 0.5 for i in range(20)])
        assert engine._order == 2

    def test_out_of_range_order_is_clamped(self) -> None:
        adaptor = HyperparameterAdaptor(redis_client=_FakeRedis())
        adaptor.save("s1", "taylor_finite_differences", {"order": 99.0})
        engine = TaylorPredictionEngine(
            order=1, horizon=1, series_id="s1",
            hyperparameter_adaptor=adaptor,
        )
        engine.predict([20.0 + i * 0.5 for i in range(20)])
        assert engine._order == 3  # clamped to max

    def test_debug_log_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        adaptor = HyperparameterAdaptor(redis_client=_FakeRedis())
        adaptor.save("s1", "taylor_finite_differences", {"order": 2.0})
        engine = TaylorPredictionEngine(
            order=1, horizon=1, series_id="s1",
            hyperparameter_adaptor=adaptor,
        )
        with caplog.at_level(
            logging.DEBUG,
            logger="iot_machine_learning.infrastructure.ml.engines.taylor.engine",
        ):
            engine.predict([20.0 + i * 0.5 for i in range(20)])
        assert any("hyperparams_loaded" in r.getMessage() for r in caplog.records)


# -- Statistical ----------------------------------------------------------


class TestStatisticalHyperparams:
    def test_alpha_beta_loaded_from_adaptor(self) -> None:
        adaptor = HyperparameterAdaptor(redis_client=_FakeRedis())
        adaptor.save(
            "s2",
            "statistical_ema_holt",
            {"alpha": 0.7, "beta": 0.25, "mae": 1.5},
        )
        engine = StatisticalPredictionEngine(
            alpha=0.3, beta=0.1, horizon=1,
            series_id="s2",
            hyperparameter_adaptor=adaptor,
        )
        engine.predict([10.0 + i * 0.2 for i in range(15)])
        assert engine._alpha == pytest.approx(0.7)
        assert engine._beta == pytest.approx(0.25)
        assert engine._current_mae == pytest.approx(1.5)

    def test_defaults_when_adaptor_absent(self) -> None:
        engine = StatisticalPredictionEngine(
            alpha=0.4, beta=0.2, horizon=1, series_id="s2",
        )
        engine.predict([10.0 + i * 0.2 for i in range(15)])
        assert engine._alpha == pytest.approx(0.4)
        assert engine._beta == pytest.approx(0.2)

    def test_invalid_stored_values_ignored(self) -> None:
        adaptor = HyperparameterAdaptor(redis_client=_FakeRedis())
        adaptor.save("s2", "statistical_ema_holt", {"alpha": 1.5, "beta": 2.0})
        engine = StatisticalPredictionEngine(
            alpha=0.3, beta=0.1, series_id="s2",
            hyperparameter_adaptor=adaptor,
        )
        engine.predict([10.0 + i * 0.2 for i in range(15)])
        # Out-of-range values are ignored; defaults preserved.
        assert engine._alpha == pytest.approx(0.3)
        assert engine._beta == pytest.approx(0.1)

    def test_debug_log_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        adaptor = HyperparameterAdaptor(redis_client=_FakeRedis())
        adaptor.save("s2", "statistical_ema_holt", {"alpha": 0.5, "beta": 0.2})
        engine = StatisticalPredictionEngine(
            alpha=0.3, beta=0.1, series_id="s2",
            hyperparameter_adaptor=adaptor,
        )
        with caplog.at_level(
            logging.DEBUG,
            logger="iot_machine_learning.infrastructure.ml.engines.statistical.engine",
        ):
            engine.predict([10.0 + i * 0.2 for i in range(15)])
        assert any("hyperparams_loaded" in r.getMessage() for r in caplog.records)


# -- Feedback loop closure ------------------------------------------------


class TestFeedbackLoopClosure:
    """Verify that hyperparameters persist across engine / pipeline instances.

    A params write via ``adaptor.save(...)`` in pipeline A must be picked
    up by a freshly constructed engine wired to the same adaptor in
    pipeline B. This closes the IMP-4c feedback loop: learned params
    outlive a single :class:`PipelineExecutor` instance.
    """

    def test_params_persist_across_engine_instances(self) -> None:
        redis = _FakeRedis()
        adaptor_a = HyperparameterAdaptor(redis_client=redis)
        adaptor_b = HyperparameterAdaptor(redis_client=redis)

        # Pipeline A learns a better alpha/beta and writes via adaptor.
        adaptor_a.save(
            "sensor-42",
            "statistical_ema_holt",
            {"alpha": 0.65, "beta": 0.22, "mae": 0.8},
        )

        # Pipeline B spins up a fresh engine sharing the same Redis but a
        # different adaptor instance, mimicking a new PipelineExecutor.
        engine_b = StatisticalPredictionEngine(
            alpha=0.3, beta=0.1, series_id="sensor-42",
            hyperparameter_adaptor=adaptor_b,
        )
        engine_b.predict([5.0 + 0.1 * i for i in range(20)])

        assert engine_b._alpha == pytest.approx(0.65)
        assert engine_b._beta == pytest.approx(0.22)
        assert engine_b._current_mae == pytest.approx(0.8)

    def test_params_persist_across_taylor_instances(self) -> None:
        redis = _FakeRedis()
        adaptor_a = HyperparameterAdaptor(redis_client=redis)
        adaptor_b = HyperparameterAdaptor(redis_client=redis)

        adaptor_a.save("sensor-7", "taylor_finite_differences", {"order": 3.0})

        engine_b = TaylorPredictionEngine(
            order=1, horizon=1, series_id="sensor-7",
            hyperparameter_adaptor=adaptor_b,
        )
        engine_b.predict([20.0 + i * 0.5 for i in range(20)])
        assert engine_b._order == 3

    def test_inert_adaptor_breaks_the_loop(self) -> None:
        """When Redis is absent, params do NOT persist — engines keep defaults."""
        adaptor_a = HyperparameterAdaptor(redis_client=None)  # inert
        adaptor_b = HyperparameterAdaptor(redis_client=None)

        adaptor_a.save("sensor-42", "statistical_ema_holt", {"alpha": 0.9})

        engine_b = StatisticalPredictionEngine(
            alpha=0.3, beta=0.1, series_id="sensor-42",
            hyperparameter_adaptor=adaptor_b,
        )
        engine_b.predict([5.0 + 0.1 * i for i in range(20)])
        assert engine_b._alpha == pytest.approx(0.3)  # default preserved
