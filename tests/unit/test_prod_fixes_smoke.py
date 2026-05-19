"""Smoke test para PROD-1, PROD-2, PROD-3.

Verifica presencia de circuit breaker, observabilidad cognitiva,
endpoints de diagnóstico, y archivos de load testing.
"""
from __future__ import annotations

import ast
import os

import pytest

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _read(path: str) -> str:
    with open(os.path.join(BASE, path), "r", encoding="utf-8") as f:
        return f.read()


def _has(source: str, *needles: str) -> bool:
    return all(n in source for n in needles)


def _line_count(path: str) -> int:
    return len(_read(path).splitlines())


class TestProd1Resilience:
    """PROD-1: Circuit breaker en TSDB y DistributedWindow."""

    def test_tsdb_adapter_has_circuit_breaker(self):
        src = _read("infrastructure/persistence/redis/tsdb_adapter.py")
        assert _has(src, "get_redis_circuit_breaker", "_circuit", "_log_transition")
        assert _has(src, "[PROD-1] tsdb_circuit_opened", "[PROD-1] tsdb_circuit_probing")

    def test_tsdb_adapter_line_limit(self):
        assert _line_count("infrastructure/persistence/redis/tsdb_adapter.py") <= 180

    def test_distributed_window_has_circuit_breaker(self):
        src = _read("infrastructure/persistence/redis/distributed_window.py")
        assert _has(src, "get_redis_circuit_breaker", "_circuit", "_log_transition")
        assert _has(src, "[PROD-1] dist_window_circuit_opened")

    def test_distributed_window_line_limit(self):
        assert _line_count("infrastructure/persistence/redis/distributed_window.py") <= 180

    def test_health_endpoint_exposes_circuit_state(self):
        src = _read("ml_service/api/routes_health.py")
        assert _has(src, "tsdb_circuit", "dist_window_circuit", "degraded")

    def test_health_response_schema_has_circuit_fields(self):
        src = _read("ml_service/api/schemas.py")
        assert _has(src, "tsdb_circuit:", "dist_window_circuit:")

    def test_circuit_factory_accepts_kwargs(self):
        src = _read("infrastructure/persistence/redis/circuit_factory.py")
        assert _has(src, "failure_threshold: int = 0", "recovery_timeout: int = 0")


class TestProd2Observability:
    """PROD-2: Observabilidad granular del cognitive orchestrator."""

    def test_pipeline_executor_has_timing(self):
        src = _read("infrastructure/ml/cognitive/orchestration/pipeline_executor.py")
        assert _has(src, "perf_counter", "_warn_ms", "slow_phase", "record_cognitive_phase")

    def test_pipeline_executor_line_limit(self):
        assert _line_count("infrastructure/ml/cognitive/orchestration/pipeline_executor.py") <= 180

    def test_performance_metrics_has_cognitive_methods(self):
        src = _read("ml_service/metrics/performance_metrics.py")
        assert _has(src, "record_cognitive_phase", "record_cognitive_budget_exceeded")
        assert _has(src, "record_cognitive_phase_skipped", "record_cognitive_fallback")

    def test_metrics_types_has_cognitive_fields(self):
        src = _read("ml_service/metrics/metrics_types.py")
        assert _has(src, "cognitive_budget_exceeded", "cognitive_phases_skipped", "cognitive_fallbacks")

    def test_prometheus_serializer_has_cognitive(self):
        src = _read("ml_service/metrics/prometheus_serializer.py")
        assert _has(src, "serialize_cognitive_prometheus", "zenin_cognitive_phase_duration_seconds")

    def test_routes_has_diagnostics_endpoint(self):
        src = _read("ml_service/api/routes.py")
        assert _has(src, "/ml/diagnostics", "slowest_phases", "cognitive_stats")


class TestProd3LoadTesting:
    """PROD-3: Locust load testing."""

    def test_locustfile_exists(self):
        assert os.path.exists(os.path.join(BASE, "tests/load/locustfile.py"))

    def test_scenario_1000_sensors_exists(self):
        assert os.path.exists(os.path.join(BASE, "tests/load/scenarios/scenario_1000_sensors.py"))

    def test_scenario_cognitive_load_exists(self):
        assert os.path.exists(os.path.join(BASE, "tests/load/scenarios/scenario_cognitive_load.py"))

    def test_scenario_redis_failover_exists(self):
        assert os.path.exists(os.path.join(BASE, "tests/load/scenarios/scenario_redis_failover.py"))

    def test_locustfile_line_limit(self):
        assert _line_count("tests/load/locustfile.py") <= 180

    def test_scenarios_line_limit(self):
        assert _line_count("tests/load/scenarios/scenario_1000_sensors.py") <= 180
        assert _line_count("tests/load/scenarios/scenario_cognitive_load.py") <= 180
        assert _line_count("tests/load/scenarios/scenario_redis_failover.py") <= 180


class TestEnvVariables:
    """Verifica variables de entorno documentadas."""

    def test_env_example_has_prod_vars(self):
        env_path = os.path.abspath(os.path.join(BASE, "..", ".env.example"))
        with open(env_path, "r", encoding="utf-8") as f:
            src = f.read()
        assert _has(src, "ML_TSDB_CIRCUIT_FAILURE_THRESHOLD", "ML_TSDB_CIRCUIT_RECOVERY_TIMEOUT_SECONDS")
        assert _has(src, "ML_COGNITIVE_PHASE_WARN_MS")
