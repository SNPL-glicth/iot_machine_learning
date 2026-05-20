"""Tests for warm-start module."""
import pytest


class TestWarmupResult:
    """Test WarmupResult data class."""

    def test_importable(self):
        from iot_machine_learning.ml_service.warmup import WarmupResult
        assert WarmupResult is not None

    def test_record_success(self):
        from iot_machine_learning.ml_service.warmup import WarmupResult
        r = WarmupResult()
        r.record("test_component", True, 5.0, "ok")
        assert r.all_healthy is True
        assert r.components["test_component"]["success"] is True

    def test_record_failure(self):
        from iot_machine_learning.ml_service.warmup import WarmupResult
        r = WarmupResult()
        r.record("healthy", True, 1.0)
        r.record("broken", False, 2.0, "connection refused")
        assert r.all_healthy is False

    def test_to_dict(self):
        from iot_machine_learning.ml_service.warmup import WarmupResult
        r = WarmupResult()
        r.record("engine", True, 10.5, "loaded")
        r.total_ms = 10.5
        d = r.to_dict()
        assert d["all_healthy"] is True
        assert d["total_ms"] == 10.5
        assert "engine" in d["components"]


class TestWarmupEngines:
    """Test engine warmup."""

    def test_warmup_engines_function_exists(self):
        from iot_machine_learning.ml_service.warmup import warmup_engines
        assert callable(warmup_engines)

    def test_warmup_engines_runs(self):
        from iot_machine_learning.ml_service.warmup import warmup_engines, WarmupResult
        result = WarmupResult()
        warmup_engines(result)
        # Taylor and Statistical should always succeed (no external deps)
        assert "taylor_engine" in result.components
        assert "statistical_engine" in result.components
        assert result.components["taylor_engine"]["success"] is True
        assert result.components["statistical_engine"]["success"] is True


class TestWarmupConfig:
    """Test config warmup."""

    def test_warmup_config_function_exists(self):
        from iot_machine_learning.ml_service.warmup import warmup_config
        assert callable(warmup_config)

    def test_warmup_config_runs(self):
        from iot_machine_learning.ml_service.warmup import warmup_config, WarmupResult
        result = WarmupResult()
        warmup_config(result)
        assert "config" in result.components
        assert result.components["config"]["success"] is True


class TestRunWarmup:
    """Test full warmup orchestration."""

    def test_run_warmup_function_exists(self):
        from iot_machine_learning.ml_service.warmup import run_warmup
        assert callable(run_warmup)

    def test_run_warmup_returns_result(self):
        from iot_machine_learning.ml_service.warmup import run_warmup
        result = run_warmup()
        assert result.total_ms > 0
        assert "taylor_engine" in result.components
        assert "config" in result.components
