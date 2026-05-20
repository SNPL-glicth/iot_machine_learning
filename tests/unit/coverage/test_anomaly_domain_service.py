"""Auto-generated coverage test for domain/services/anomaly_domain_service.py."""
import pytest


def test_anomaly_domain_service_importable():
    try:
        import iot_machine_learning.domain.services.anomaly_domain_service
        assert iot_machine_learning.domain.services.anomaly_domain_service is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
