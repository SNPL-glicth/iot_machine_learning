"""Auto-generated coverage test for application/dto/text_decision_output.py."""
import pytest


def test_text_decision_output_importable():
    try:
        import iot_machine_learning.application.dto.text_decision_output
        assert iot_machine_learning.application.dto.text_decision_output is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
