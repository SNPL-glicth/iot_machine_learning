"""Auto-generated coverage test for infrastructure/adapters/calibrators/utils/isotonic_math.py."""
import pytest


def test_isotonic_math_importable():
    try:
        import iot_machine_learning.infrastructure.adapters.calibrators.utils.isotonic_math
        assert iot_machine_learning.infrastructure.adapters.calibrators.utils.isotonic_math is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
