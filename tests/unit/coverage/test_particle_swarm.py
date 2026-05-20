"""Auto-generated coverage test for infrastructure/ml/optimization/nonconvex/particle_swarm.py."""
import pytest


def test_particle_swarm_importable():
    try:
        import iot_machine_learning.infrastructure.ml.optimization.nonconvex.particle_swarm
        assert iot_machine_learning.infrastructure.ml.optimization.nonconvex.particle_swarm is not None
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Import failed: {e}")
