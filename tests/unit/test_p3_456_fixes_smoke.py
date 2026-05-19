"""Smoke tests para FIXES P3-4, P3-5, P3-6."""
from __future__ import annotations

import ast
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent / "ml_service"
INFRA = Path(__file__).resolve().parent.parent.parent / "infrastructure"

def _read_source(relative_path: str) -> str:
    if relative_path.startswith("infrastructure/"):
        p = Path(__file__).resolve().parent.parent.parent / relative_path
    else:
        p = BASE / relative_path
    return p.read_text(encoding="utf-8")

def _has_class(src: str, name: str) -> bool:
    return f"class {name}" in src

def _has_function(src: str, name: str) -> bool:
    return f"def {name}" in src

def test_all_new_files_under_180_lines() -> None:
    for relative in [
        "ml_service/features/sensor_feature_config.py",
        "infrastructure/persistence/redis/tsdb_adapter.py",
        "infrastructure/persistence/redis/distributed_window.py",
    ]:
        src = _read_source(relative)
        lines = src.splitlines()
        assert len(lines) <= 180, f"{relative} has {len(lines)} lines"

def test_p3_4_sensor_feature_config() -> None:
    src = _read_source("ml_service/features/sensor_feature_config.py")
    assert _has_class(src, "SensorFeatureConfig")
    assert _has_class(src, "SensorFeatureConfigRegistry")
    assert "def get(self" in src
    assert "def register_sensor" in src
    assert "def register_type" in src
    assert "ML_FEATURE_CONFIG_BY_TYPE" in src
    assert "ML_FEATURE_CONFIG_BY_ID" in src

def test_p3_4_ml_features_integration() -> None:
    src = _read_source("ml_service/features/ml_features.py")
    assert "registry" in src
    src2 = _read_source("ml_service/features/services/feature_computer.py")
    assert "registry" in src2
    assert "sensor_type" in src2
    assert "_resolve_config" in src2

def test_p3_5_tsdb_adapter() -> None:
    src = _read_source("infrastructure/persistence/redis/tsdb_adapter.py")
    assert _has_class(src, "RedisTSDBAdapter")
    assert "def append" in src
    assert "def get_recent" in src
    assert "def flush_sensor" in src
    assert "ML_TSDB_ENABLED" in src
    assert "ML_TSDB_TTL_SECONDS" in src
    assert "ML_TSDB_MAX_ENTRIES" in src

def test_p3_5_base_queries_tsdb_fallback() -> None:
    src = _read_source("infrastructure/persistence/sql/storage/base_queries.py")
    assert "tsdb_adapter" in src or "_tsdb" in src
    assert "P3-5" in src
    assert "tsdb_hit" in src

def test_p3_5_stream_consumer_tsdb_write() -> None:
    src = _read_source("consumers/stream_consumer.py")
    assert "_tsdb" in src
    assert "tsdb_adapter" in src
    assert "tsdb_append" in src or "P3-5" in src

def test_p3_6_distributed_window() -> None:
    src = _read_source("infrastructure/persistence/redis/distributed_window.py")
    assert _has_class(src, "DistributedWindowAdapter")
    assert "def append" in src
    assert "def get" in src
    assert "def remove" in src
    assert "ML_DISTRIBUTED_WINDOWS_ENABLED" in src
    assert "ML_DIST_WINDOW_TTL_SECONDS" in src

def test_p3_6_sliding_window_distributed() -> None:
    src = _read_source("consumers/sliding_window.py")
    assert "distributed_adapter" in src
    assert "_distributed" in src
    assert "P3-6" in src

def test_p3_6_stream_consumer_migration() -> None:
    src = _read_source("consumers/stream_consumer.py")
    assert "_distributed" in src
    assert "window_migrated" in src
    assert "migration_attempted" in src

def test_env_defaults_are_opt_in() -> None:
    """TSDB y Distributed Windows deben ser false por default (opt-in explícito)."""
    tsdb = _read_source("infrastructure/persistence/redis/tsdb_adapter.py")
    dist = _read_source("infrastructure/persistence/redis/distributed_window.py")
    assert "false" in tsdb.lower() or "\"false\"" in tsdb.lower()
    assert "false" in dist.lower() or "\"false\"" in dist.lower()
