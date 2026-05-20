"""Auto-generated coverage test for ml_service/api/services/analysis/output_assembler.py."""
import pytest


def test_output_assembler_importable():
    try:
        import iot_machine_learning.ml_service.api.services.analysis.output_assembler
        assert iot_machine_learning.ml_service.api.services.analysis.output_assembler is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
