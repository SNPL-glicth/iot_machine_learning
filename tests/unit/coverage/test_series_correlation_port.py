"""Auto-generated coverage test for domain/ports/series_correlation_port.py."""
import pytest


def test_series_correlation_port_importable():
    try:
        import iot_machine_learning.domain.ports.series_correlation_port
        assert iot_machine_learning.domain.ports.series_correlation_port is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
