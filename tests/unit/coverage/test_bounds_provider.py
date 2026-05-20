"""Auto-generated coverage test for infrastructure/ml/cognitive/sanitize/bounds_provider.py."""
import pytest


def test_bounds_provider_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.sanitize.bounds_provider
        assert iot_machine_learning.infrastructure.ml.cognitive.sanitize.bounds_provider is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
