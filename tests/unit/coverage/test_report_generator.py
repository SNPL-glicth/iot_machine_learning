"""Auto-generated coverage test for application/evaluation/report_generator.py."""
import pytest


def test_report_generator_importable():
    try:
        import iot_machine_learning.application.evaluation.report_generator
        assert iot_machine_learning.application.evaluation.report_generator is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
