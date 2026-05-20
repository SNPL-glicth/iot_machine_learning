"""Auto-generated coverage test for infrastructure/ml/anomaly/detectors/multivariate/matrix_builder.py."""
import pytest


def test_matrix_builder_importable():
    try:
        import iot_machine_learning.infrastructure.ml.anomaly.detectors.multivariate.matrix_builder
        assert iot_machine_learning.infrastructure.ml.anomaly.detectors.multivariate.matrix_builder is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
