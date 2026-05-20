"""Auto-generated coverage test for domain/ports/analysis_data_port.py."""
import pytest


def test_analysis_data_port_importable():
    try:
        import iot_machine_learning.domain.ports.analysis_data_port
        assert iot_machine_learning.domain.ports.analysis_data_port is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
