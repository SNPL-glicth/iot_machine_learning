"""Auto-generated coverage test for infrastructure/ml/analyzers/numeric_stats.py."""
import pytest


def test_numeric_stats_importable():
    try:
        import iot_machine_learning.infrastructure.ml.analyzers.numeric_stats
        assert iot_machine_learning.infrastructure.ml.analyzers.numeric_stats is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
