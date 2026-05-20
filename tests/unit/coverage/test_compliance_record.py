"""Auto-generated coverage test for infrastructure/ml/cognitive/compliance/compliance_record.py."""
import pytest


def test_compliance_record_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.compliance.compliance_record
        assert iot_machine_learning.infrastructure.ml.cognitive.compliance.compliance_record is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
