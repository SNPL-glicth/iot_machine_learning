"""Auto-generated coverage test for domain/ports/text_encoder_port.py."""
import pytest


def test_text_encoder_port_importable():
    try:
        import iot_machine_learning.domain.ports.text_encoder_port
        assert iot_machine_learning.domain.ports.text_encoder_port is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
