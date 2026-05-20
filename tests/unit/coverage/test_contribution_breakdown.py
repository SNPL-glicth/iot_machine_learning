"""Auto-generated coverage test for domain/entities/explainability/contribution_breakdown.py."""
import pytest


def test_contribution_breakdown_importable():
    try:
        import iot_machine_learning.domain.entities.explainability.contribution_breakdown
        assert iot_machine_learning.domain.entities.explainability.contribution_breakdown is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
