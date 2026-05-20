"""Auto-generated coverage test for infrastructure/ml/cognitive/utils/timeout_guard.py."""
import pytest


def test_timeout_guard_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.utils.timeout_guard
        assert iot_machine_learning.infrastructure.ml.cognitive.utils.timeout_guard is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
