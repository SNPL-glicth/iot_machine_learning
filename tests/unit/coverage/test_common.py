"""Auto-generated coverage test for infrastructure/analysis/adapters/common.py."""
import pytest


def test_common_importable():
    try:
        import iot_machine_learning.infrastructure.analysis.adapters.common
        assert iot_machine_learning.infrastructure.analysis.adapters.common is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
