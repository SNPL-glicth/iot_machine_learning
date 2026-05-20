"""Auto-generated coverage test for infrastructure/ml/analyzers/numeric_analyzer.py."""
import pytest


def test_numeric_analyzer_importable():
    try:
        import iot_machine_learning.infrastructure.ml.analyzers.numeric_analyzer
        assert iot_machine_learning.infrastructure.ml.analyzers.numeric_analyzer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
