"""Tests para sincronización BayesianWeightTracker → VotingStrategy."""
import pytest

from infrastructure.ml.anomaly.voting.strategy import VotingStrategy
from core.ensemble.ensemble_drift_coupling import EnsembleWeightState


def test_sync_with_bayesian_tracker():
    """VotingStrategy debe sincronizar pesos con BayesianWeightTracker."""
    strategy = VotingStrategy(
        weights={"a": 0.5, "b": 0.5},
        threshold=0.5,
    )
    
    # Simular pesos del BayesianWeightTracker
    tracker_weights = {"a": 0.7, "b": 0.3}
    
    strategy._sync_with_bayesian_tracker(tracker_weights)
    
    # Pesos deben haberse actualizado
    assert strategy._weight_state.current_weights == tracker_weights


def test_sync_hysteresis():
    """Sincronización debe aplicar hysteresis."""
    strategy = VotingStrategy(
        weights={"a": 0.5, "b": 0.5},
        threshold=0.5,
    )
    
    # Cambio pequeño (< 15%) → no actualizar
    small_change = {"a": 0.52, "b": 0.48}
    strategy._sync_with_bayesian_tracker(small_change)
    
    assert strategy._weight_state.current_weights == {"a": 0.5, "b": 0.5}
    
    # Cambio grande (> 15%) → actualizar
    large_change = {"a": 0.7, "b": 0.3}
    strategy._sync_with_bayesian_tracker(large_change)
    
    assert strategy._weight_state.current_weights == large_change


def test_sync_updates_drift_coupling():
    """Sincronización debe actualizar drift coupling si está activo."""
    from core.ensemble.ensemble_drift_coupling import EnsembleDriftCoupling
    import numpy as np
    
    detectors = {
        "a": type("MockDetector", (), {"detect": lambda x: np.array([0.5] * len(x))})(),
        "b": type("MockDetector", (), {"detect": lambda x: np.array([0.5] * len(x))})(),
    }
    
    # Use 1000 samples to meet min_samples requirement
    coupling = EnsembleDriftCoupling(
        detectors=detectors,
        calibration_data=np.random.randn(1000),
        initial_weights={"a": 0.5, "b": 0.5},
    )
    
    strategy = VotingStrategy(
        weights={"a": 0.5, "b": 0.5},
        threshold=0.5,
        drift_coupling=coupling,
    )
    
    tracker_weights = {"a": 0.8, "b": 0.2}
    strategy._sync_with_bayesian_tracker(tracker_weights)
    
    # Ambos estados deben estar actualizados
    assert strategy._weight_state.current_weights == tracker_weights
    assert coupling.weight_state.current_weights == tracker_weights


def test_voting_strategy_uses_drift_coupling_weights():
    """VotingStrategy debe usar pesos del drift coupling cuando está activo."""
    from core.ensemble.ensemble_drift_coupling import EnsembleDriftCoupling
    import numpy as np
    
    detectors = {
        "a": type("MockDetector", (), {"detect": lambda x: np.array([0.5] * len(x))})(),
        "b": type("MockDetector", (), {"detect": lambda x: np.array([0.5] * len(x))})(),
    }
    
    # Use 1000 samples to meet min_samples requirement
    coupling = EnsembleDriftCoupling(
        detectors=detectors,
        calibration_data=np.random.randn(1000),
        initial_weights={"a": 0.3, "b": 0.7},
    )
    
    strategy = VotingStrategy(
        weights={"a": 0.5, "b": 0.5},  # Pesos originales
        threshold=0.5,
        drift_coupling=coupling,
    )
    
    # _get_effective_weights debe usar pesos del coupling
    effective = strategy._get_effective_weights()
    assert effective == {"a": 0.3, "b": 0.7}


def test_voting_strategy_without_drift_coupling():
    """VotingStrategy sin drift coupling debe usar pesos originales."""
    strategy = VotingStrategy(
        weights={"a": 0.5, "b": 0.5},
        threshold=0.5,
    )
    
    effective = strategy._get_effective_weights()
    assert effective == {"a": 0.5, "b": 0.5}


def test_voting_strategy_calibrated_weights_priority():
    """Pesos calibrados tienen prioridad sobre originales."""
    strategy = VotingStrategy(
        weights={"a": 0.5, "b": 0.5},
        threshold=0.5,
        use_calibrated_weights=True,
    )
    
    # Simular calibración
    strategy._calibrated_weights = {"a": 0.3, "b": 0.7}
    
    effective = strategy._get_effective_weights()
    assert effective == {"a": 0.3, "b": 0.7}


def test_drift_coupling_priority_over_calibrated():
    """Drift coupling tiene prioridad sobre pesos calibrados."""
    from core.ensemble.ensemble_drift_coupling import EnsembleDriftCoupling
    import numpy as np
    
    detectors = {
        "a": type("MockDetector", (), {"detect": lambda x: np.array([0.5] * len(x))})(),
        "b": type("MockDetector", (), {"detect": lambda x: np.array([0.5] * len(x))})(),
    }
    
    # Use 1000 samples to meet min_samples requirement
    coupling = EnsembleDriftCoupling(
        detectors=detectors,
        calibration_data=np.random.randn(1000),
        initial_weights={"a": 0.1, "b": 0.9},
    )
    
    strategy = VotingStrategy(
        weights={"a": 0.5, "b": 0.5},
        threshold=0.5,
        use_calibrated_weights=True,
        drift_coupling=coupling,
    )
    
    strategy._calibrated_weights = {"a": 0.3, "b": 0.7}
    
    effective = strategy._get_effective_weights()
    # Drift coupling tiene prioridad
    assert effective == {"a": 0.1, "b": 0.9}


def test_weight_state_history():
    """Weight state debe mantener historia de cambios."""
    state = EnsembleWeightState()
    
    state.update({"a": 0.5, "b": 0.5})
    state.update({"a": 0.6, "b": 0.4})
    state.update({"a": 0.7, "b": 0.3})
    
    assert len(state.weight_history) == 3
    assert state.weight_history[0] == {"a": 0.5, "b": 0.5}
    assert state.weight_history[2] == {"a": 0.7, "b": 0.3}
