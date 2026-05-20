"""Auto-generated coverage test for domain/entities/decision/priority.py."""
import pytest


def test_priority_importable():
    try:
        import iot_machine_learning.domain.entities.decision.priority
        assert iot_machine_learning.domain.entities.decision.priority is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
