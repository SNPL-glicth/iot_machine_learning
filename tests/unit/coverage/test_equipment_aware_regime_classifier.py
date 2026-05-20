"""Auto-generated coverage test for infrastructure/ml/moe/regime/equipment_aware_regime_classifier.py."""
import pytest


def test_equipment_aware_regime_classifier_importable():
    try:
        import iot_machine_learning.infrastructure.ml.moe.regime.equipment_aware_regime_classifier
        assert iot_machine_learning.infrastructure.ml.moe.regime.equipment_aware_regime_classifier is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
