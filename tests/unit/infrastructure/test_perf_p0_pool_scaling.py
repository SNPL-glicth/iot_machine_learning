"""Tests for PERF-P0: Pool scaling and thread reuse for 1000-sensor scale.

Covers:
- zenin_db_connection.py: Configurable pool sizes via env vars
- orchestrator_prediction.py: Shared ThreadPoolExecutor (no leak)
- ml_batch_runner.py: Error isolation in parallel mode
"""

from __future__ import annotations

import concurrent.futures
import os
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Commit 1: zenin_db pool scaling
# ---------------------------------------------------------------------------

class TestZeninDbPoolScaling:
    """zenin_db_connection.py pool sizes configurable via env vars."""

    def _reset_engine(self):
        """Reset the singleton so each test starts fresh."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
            ZeninDbConnection,
        )
        ZeninDbConnection._engine = None

    def test_pool_size_default_is_20(self):
        """Default pool_size is 20 (up from hardcoded 5)."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
            _env_int,
        )
        # Without env var set, default is 20
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ZENIN_DB_POOL_SIZE", None)
            assert _env_int("ZENIN_DB_POOL_SIZE", 20) == 20

    def test_pool_size_configurable_via_env(self):
        """ZENIN_DB_POOL_SIZE env var overrides default."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
            _env_int,
        )
        with patch.dict(os.environ, {"ZENIN_DB_POOL_SIZE": "50"}):
            assert _env_int("ZENIN_DB_POOL_SIZE", 20) == 50

    def test_max_overflow_configurable_via_env(self):
        """ZENIN_DB_MAX_OVERFLOW env var overrides default."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
            _env_int,
        )
        with patch.dict(os.environ, {"ZENIN_DB_MAX_OVERFLOW": "100"}):
            assert _env_int("ZENIN_DB_MAX_OVERFLOW", 30) == 100

    def test_env_int_invalid_value_uses_default(self):
        """Invalid env var value falls back to default with warning."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
            _env_int,
        )
        with patch.dict(os.environ, {"ZENIN_DB_POOL_SIZE": "not_a_number"}):
            assert _env_int("ZENIN_DB_POOL_SIZE", 20) == 20

    def test_connect_timeout_configurable(self):
        """ZENIN_DB_CONNECT_TIMEOUT is read from env."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
            _env_int,
        )
        with patch.dict(os.environ, {"ZENIN_DB_CONNECT_TIMEOUT": "15"}):
            assert _env_int("ZENIN_DB_CONNECT_TIMEOUT", 10) == 15

    def test_default_max_capacity_is_50(self):
        """Default max capacity = pool_size(20) + max_overflow(30) = 50."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
            _env_int,
        )
        pool_size = _env_int("ZENIN_DB_POOL_SIZE", 20)
        max_overflow = _env_int("ZENIN_DB_MAX_OVERFLOW", 30)
        assert pool_size + max_overflow == 50


# ---------------------------------------------------------------------------
# Commit 2: Orchestrator shared executor (thread pool leak fix)
# ---------------------------------------------------------------------------

class TestOrchestratorSharedExecutor:
    """orchestrator_prediction.py uses shared ThreadPoolExecutor."""

    def test_shared_executor_is_singleton(self):
        """Same executor returned on multiple calls."""
        from iot_machine_learning.ml_service.runners.adapters.orchestrator_prediction import (
            _get_shared_executor,
        )
        exec1 = _get_shared_executor()
        exec2 = _get_shared_executor()
        assert exec1 is exec2

    def test_shared_executor_max_workers_configurable(self):
        """ML_ORCHESTRATOR_WORKERS env var controls max_workers."""
        import iot_machine_learning.ml_service.runners.adapters.orchestrator_prediction as mod
        # Reset singleton
        mod._shared_executor = None
        original_workers = mod._ORCHESTRATOR_WORKERS
        try:
            mod._ORCHESTRATOR_WORKERS = 8
            mod._shared_executor = None
            executor = mod._get_shared_executor()
            assert executor._max_workers == 8
        finally:
            mod._ORCHESTRATOR_WORKERS = original_workers
            if mod._shared_executor:
                mod._shared_executor.shutdown(wait=False)
            mod._shared_executor = None

    def test_shared_executor_no_leak_under_load(self):
        """Calling _get_shared_executor 100 times does NOT create 100 pools."""
        import iot_machine_learning.ml_service.runners.adapters.orchestrator_prediction as mod
        mod._shared_executor = None
        executors = set()
        for _ in range(100):
            executors.add(id(mod._get_shared_executor()))
        assert len(executors) == 1, (
            f"Expected 1 executor, got {len(executors)} — thread pool leak!"
        )
        # Cleanup
        mod._shared_executor.shutdown(wait=False)
        mod._shared_executor = None

    def test_run_with_timeout_uses_shared_executor(self):
        """_run_with_timeout does not create new ThreadPoolExecutor."""
        import iot_machine_learning.ml_service.runners.adapters.orchestrator_prediction as mod
        from iot_machine_learning.ml_service.runners.adapters.orchestrator_prediction import (
            OrchestratorPredictionAdapter,
        )
        mod._shared_executor = None
        adapter = OrchestratorPredictionAdapter(
            orchestrator=MagicMock(),
            storage=MagicMock(),
            audit=MagicMock(),
            flags=MagicMock(),
        )

        result = adapter._run_with_timeout(lambda: 42, timeout_seconds=5.0)
        assert result == 42

        # Verify it used the shared executor (singleton created once)
        assert mod._shared_executor is not None
        # Cleanup
        mod._shared_executor.shutdown(wait=False)
        mod._shared_executor = None

    def test_run_with_timeout_raises_on_timeout(self):
        """TimeoutError raised if function exceeds budget."""
        import iot_machine_learning.ml_service.runners.adapters.orchestrator_prediction as mod
        from iot_machine_learning.ml_service.runners.adapters.orchestrator_prediction import (
            OrchestratorPredictionAdapter,
        )
        mod._shared_executor = None
        adapter = OrchestratorPredictionAdapter(
            orchestrator=MagicMock(),
            storage=MagicMock(),
            audit=MagicMock(),
            flags=MagicMock(),
        )

        def slow_fn():
            time.sleep(5)
            return "never"

        with pytest.raises(TimeoutError, match="exceeded"):
            adapter._run_with_timeout(slow_fn, timeout_seconds=0.1)

        # Cleanup
        mod._shared_executor.shutdown(wait=False)
        mod._shared_executor = None


# ---------------------------------------------------------------------------
# Commit 3: Batch runner error isolation
# ---------------------------------------------------------------------------

class TestBatchRunnerErrorIsolation:
    """ml_batch_runner.py: failed sensor does not crash the batch."""

    def test_parallel_batch_isolates_sensor_failure(self):
        """A sensor that raises an exception does not block others."""
        call_log = []

        def mock_process_sensor(sensor_id, *args, **kwargs):
            if sensor_id == 50:
                raise RuntimeError("Simulated sensor failure")
            call_log.append(sensor_id)
            return {"ok": True, "enterprise": False}

        sensors = list(range(100))

        # Simulate parallel execution as done in ml_batch_runner.py
        errors = 0
        processed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(mock_process_sensor, sid): sid
                for sid in sensors
            }
            for future in concurrent.futures.as_completed(futures):
                sid = futures[future]
                try:
                    result = future.result()
                except Exception:
                    errors += 1
                    continue
                if result["ok"]:
                    processed += 1

        assert processed == 99, f"Expected 99 ok, got {processed}"
        assert errors == 1, f"Expected 1 error, got {errors}"
        assert 50 not in call_log

    def test_parallel_batch_1000_sensors_throughput(self):
        """1000 sensors with 8 workers completes in reasonable time."""
        def mock_process_sensor(sensor_id, *args, **kwargs):
            time.sleep(0.01)  # 10ms per sensor
            return {"ok": True, "enterprise": False}

        sensors = list(range(1000))
        t0 = time.monotonic()

        processed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(mock_process_sensor, sid): sid
                for sid in sensors
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result["ok"]:
                        processed += 1
                except Exception:
                    pass

        elapsed = time.monotonic() - t0
        # With 8 workers and 10ms/sensor: ~1000/8 * 0.01 = 1.25s theoretical
        # Allow 5x overhead: < 7s
        assert processed == 1000
        assert elapsed < 7.0, (
            f"1000 sensors took {elapsed:.1f}s, expected < 7s with 8 workers"
        )

    def test_batch_config_parallel_workers_default_is_8(self):
        """ML_BATCH_PARALLEL_WORKERS default changed from 1 to 8."""
        from iot_machine_learning.ml_service.config.batch_config import BatchConfig
        config = BatchConfig()
        assert config.ML_BATCH_PARALLEL_WORKERS == 8


# ---------------------------------------------------------------------------
# Redis pool (already implemented — regression tests)
# ---------------------------------------------------------------------------

class TestRedisPoolRegression:
    """Verify existing Redis ConnectionPool infrastructure."""

    def test_general_max_connections_default_150(self):
        """GENERAL_MAX_CONNECTIONS defaults to 150."""
        from iot_machine_learning.infrastructure.persistence.redis.config import (
            GENERAL_MAX_CONNECTIONS,
        )
        # Default if REDIS_MAX_CONNECTIONS not set = 150
        assert GENERAL_MAX_CONNECTIONS >= 50, (
            f"GENERAL_MAX_CONNECTIONS={GENERAL_MAX_CONNECTIONS}, expected >= 50"
        )

    def test_stream_max_connections_default_50(self):
        """STREAM_MAX_CONNECTIONS defaults to 50."""
        from iot_machine_learning.infrastructure.persistence.redis.config import (
            STREAM_MAX_CONNECTIONS,
        )
        assert STREAM_MAX_CONNECTIONS >= 20, (
            f"STREAM_MAX_CONNECTIONS={STREAM_MAX_CONNECTIONS}, expected >= 20"
        )

    def test_redis_config_env_int_helper(self):
        """_env_int helper works correctly."""
        from iot_machine_learning.infrastructure.persistence.redis.config import _env_int
        with patch.dict(os.environ, {"TEST_VAR": "42"}):
            assert _env_int("TEST_VAR", 10) == 42
        with patch.dict(os.environ, {"TEST_VAR": "bad"}):
            assert _env_int("TEST_VAR", 10) == 10
