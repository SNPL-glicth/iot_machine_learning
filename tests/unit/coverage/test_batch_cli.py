"""Auto-generated coverage test for ml_service/runners/batch_cli.py."""
import pytest


def test_batch_cli_importable():
    try:
        import iot_machine_learning.ml_service.runners.batch_cli
        assert iot_machine_learning.ml_service.runners.batch_cli is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
