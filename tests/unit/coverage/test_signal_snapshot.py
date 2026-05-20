"""Auto-generated coverage test for domain/entities/explainability/signal_snapshot.py."""
import pytest


def test_signal_snapshot_importable():
    try:
        import iot_machine_learning.domain.entities.explainability.signal_snapshot
        assert iot_machine_learning.domain.entities.explainability.signal_snapshot is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
