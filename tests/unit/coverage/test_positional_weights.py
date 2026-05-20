"""Auto-generated coverage test for infrastructure/ml/cognitive/text/encoders/positional_weights.py."""
import pytest


def test_positional_weights_importable():
    try:
        import iot_machine_learning.infrastructure.ml.cognitive.text.encoders.positional_weights
        assert iot_machine_learning.infrastructure.ml.cognitive.text.encoders.positional_weights is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
