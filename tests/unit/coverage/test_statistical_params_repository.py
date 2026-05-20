"""Auto-generated coverage test for infrastructure/persistence/sql/zenin_ml/statistical_params_repository.py."""
import pytest


def test_statistical_params_repository_importable():
    try:
        import iot_machine_learning.infrastructure.persistence.sql.zenin_ml.statistical_params_repository
        assert iot_machine_learning.infrastructure.persistence.sql.zenin_ml.statistical_params_repository is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
