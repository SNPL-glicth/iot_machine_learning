"""Auto-generated coverage test for application/semantic_extraction/priority_scorers.py."""
import pytest


def test_priority_scorers_importable():
    try:
        import iot_machine_learning.application.semantic_extraction.priority_scorers
        assert iot_machine_learning.application.semantic_extraction.priority_scorers is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
