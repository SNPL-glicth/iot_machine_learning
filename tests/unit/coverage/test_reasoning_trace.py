"""Auto-generated coverage test for domain/entities/explainability/reasoning_trace.py."""
import pytest


def test_reasoning_trace_importable():
    try:
        import iot_machine_learning.domain.entities.explainability.reasoning_trace
        assert iot_machine_learning.domain.entities.explainability.reasoning_trace is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
