"""Smoke test for P3 architectural redesign fixes.

Validates statically that all P3 components exist and expose the required API.
"""
from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent / "ml_service"


def _read_source(module_path: str) -> str:
    return (BASE / module_path).read_text()


def _has_class(source: str, class_name: str) -> bool:
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.ClassDef) and node.name == class_name
        for node in ast.walk(tree)
    )


def _has_function(source: str, func_name: str) -> bool:
    tree = ast.parse(source)
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == func_name
        for node in ast.walk(tree)
    )


def test_p3_1_prediction_worker_exists():
    src = _read_source("consumers/prediction_worker.py")
    assert _has_class(src, "PredictionWorker")
    assert "ML_PREDICTION_WORKERS" in src
    assert "ML_PREDICTION_QUEUE_MAX" in src
    assert "worker_factory" in src
    assert "is_healthy" in src


def test_p3_1_stream_consumer_uses_prediction_worker():
    src = _read_source("consumers/stream_consumer.py")
    assert "PredictionWorker" in src
    assert "prediction_worker" in src
    # XACK happens before waiting for prediction result
    assert "enqueue" in src


def test_p3_2_sliding_window_sharded():
    src = _read_source("consumers/sliding_window.py")
    assert "ML_WINDOW_STORE_SHARDS" in src
    assert "_Shard" in src
    assert "n_shards" in src
    # Each shard has its own lock (shard.lock used in with statements)
    assert "shard.lock" in src


def test_p3_3_result_store_exists():
    src = _read_source("api/result_store.py")
    assert _has_class(src, "PredictionResultStore")
    assert "ML_RESULT_STORE_MAX_ENTRIES" in src
    assert "ML_RESULT_STORE_TTL_SECONDS" in src
    assert "set" in src
    assert "get" in src
    assert "is_pending" in src


def test_p3_3_routes_has_async_status_endpoint():
    src = _read_source("api/routes.py")
    assert '"/ml/predict/{prediction_id}/status"' in src
    assert "ML_ASYNC_PREDICTIONS_ENABLED" in src or "routes_predict_async" in src


def test_p3_3_routes_predict_async_exists():
    src = _read_source("api/routes_predict_async.py")
    assert "_handle_async_prediction" in src
    assert "prediction_id" in src


def test_p3_lifecycle_has_init_and_stop():
    src = _read_source("consumers/prediction_lifecycle.py")
    assert "init_prediction_worker" in src
    assert "stop_prediction_worker" in src


def test_p3_main_has_prediction_worker_stop():
    src = _read_source("lifespan.py")
    assert "prediction_worker" in src
    assert "stop_prediction_worker" in src


def test_all_new_files_under_180_lines():
    files = [
        BASE / "consumers/prediction_worker.py",
        BASE / "api/result_store.py",
        BASE / "consumers/prediction_lifecycle.py",
        BASE / "api/routes_predict_async.py",
        BASE / "consumers/sliding_window.py",
        BASE / "api/routes.py",
        BASE / "lifespan.py",
    ]
    for f in files:
        with open(f) as fh:
            lines = sum(1 for _ in fh)
        assert lines <= 180, f"{f.name} has {lines} lines (>180)"
