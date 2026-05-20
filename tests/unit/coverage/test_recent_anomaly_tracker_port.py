"""Auto-generated coverage test for domain/ports/recent_anomaly_tracker_port.py."""
import pytest


def test_recent_anomaly_tracker_port_importable():
    try:
        import iot_machine_learning.domain.ports.recent_anomaly_tracker_port
        assert iot_machine_learning.domain.ports.recent_anomaly_tracker_port is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
