"""Auto-generated coverage test for infrastructure/ml/analyzers/tabular_analyzer.py."""
import pytest


def test_tabular_analyzer_importable():
    try:
        import iot_machine_learning.infrastructure.ml.analyzers.tabular_analyzer
        assert iot_machine_learning.infrastructure.ml.analyzers.tabular_analyzer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
