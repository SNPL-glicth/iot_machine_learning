"""Unit tests for IoT Persistence Adapters (Fase 5.2).

Tests TelemetryStoragePort implementation in ml_state_store_adapter
and SeveritySqlAdapter threshold operations.
"""

from unittest.mock import MagicMock
import pytest

from infrastructure.adapters.iot.ml_state_store_adapter import (
    InMemoryMLStateStore,
    RedisMLStateStore,
    create_state_store,
)
from infrastructure.adapters.iot.severity_sql_adapter import SeveritySqlAdapter
from domain.ports.iot_ports import TelemetryStoragePort
from infrastructure.ml.cognitive.severity_classifier import (
    SeverityClassifier,
    ThresholdProvider,
)


def test_in_memory_state_store_contract():
    store = InMemoryMLStateStore()
    assert isinstance(store, TelemetryStoragePort)
    assert store.save_snapshot("test_eng", {"v": 1}) is True
    assert store.load_snapshot("test_eng") == {"v": 1}
    assert store.delete_snapshot("test_eng") is True
    assert store.load_snapshot("test_eng") is None


def test_in_memory_legacy_aliases():
    store = InMemoryMLStateStore()
    assert store.save("test_eng", {"v": 2}) is True
    assert store.load("test_eng") == {"v": 2}
    assert store.delete("test_eng") is True
    assert store.load("test_eng") is None


def test_create_state_store_factory():
    assert create_state_store(None) is None
    mem = create_state_store({"backend": "memory"})
    assert isinstance(mem, InMemoryMLStateStore)

    with pytest.raises(ValueError, match="redis backend requires 'redis_url'"):
        create_state_store({"backend": "redis"})

    with pytest.raises(ValueError, match="Unknown ML state store backend"):
        create_state_store({"backend": "cassandra"})


def test_severity_sql_adapter_user_defined_range():
    adapter = SeveritySqlAdapter()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = (10.0, 40.0)

    user_range = adapter.get_user_defined_range(mock_conn, 1)
    assert user_range == (10.0, 40.0)

    # Empty result
    mock_conn.execute.return_value.fetchone.return_value = None
    assert adapter.get_user_defined_range(mock_conn, 99) is None


def test_severity_sql_adapter_is_value_within():
    adapter = SeveritySqlAdapter()
    mock_conn = MagicMock()

    mock_conn.execute.return_value.fetchone.return_value = (15.0, 35.0)
    assert adapter.is_value_within_user_thresholds(mock_conn, 1, 20.0) is True
    assert adapter.is_value_within_user_thresholds(mock_conn, 1, 10.0) is False
    assert adapter.is_value_within_user_thresholds(mock_conn, 1, 40.0) is False


def test_severity_classifier_delegates_to_provider():
    adapter = SeveritySqlAdapter()
    assert isinstance(adapter, ThresholdProvider)
    classifier = SeverityClassifier(threshold_provider=adapter)

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = (5.0, 25.0)

    assert classifier.get_user_defined_range(mock_conn, 1) == (5.0, 25.0)
    assert classifier.is_value_within_user_thresholds(mock_conn, 1, 15.0) is True
    assert classifier.is_value_within_user_thresholds(mock_conn, 1, 30.0) is False


def test_severity_classifier_without_provider():
    classifier = SeverityClassifier()
    assert classifier.get_user_defined_range(None, 1) is None
    assert classifier.is_value_within_user_thresholds(None, 1, 99.0) is True
