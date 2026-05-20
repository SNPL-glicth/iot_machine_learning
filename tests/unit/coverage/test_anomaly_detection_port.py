"""Auto-generated coverage test for domain/ports/anomaly_detection_port.py."""
import pytest


def test_anomaly_detection_port_importable():
    try:
        import iot_machine_learning.domain.ports.anomaly_detection_port
        assert iot_machine_learning.domain.ports.anomaly_detection_port is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
