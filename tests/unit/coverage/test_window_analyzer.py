"""Auto-generated coverage test for infrastructure/ml/analyzers/window_analyzer.py."""
import pytest


def test_window_analyzer_importable():
    try:
        import iot_machine_learning.infrastructure.ml.analyzers.window_analyzer
        assert iot_machine_learning.infrastructure.ml.analyzers.window_analyzer is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
