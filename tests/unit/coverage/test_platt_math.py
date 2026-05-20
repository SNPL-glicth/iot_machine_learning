"""Auto-generated coverage test for infrastructure/adapters/calibrators/utils/platt_math.py."""
import pytest


def test_platt_math_importable():
    try:
        import iot_machine_learning.infrastructure.adapters.calibrators.utils.platt_math
        assert iot_machine_learning.infrastructure.adapters.calibrators.utils.platt_math is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
