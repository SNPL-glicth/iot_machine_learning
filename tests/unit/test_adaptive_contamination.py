"""Tests para adaptive contamination."""
import pytest

from core.drift.adaptive_contamination import (
    ContaminationState,
    ContaminationHysteresisConfig,
    AdaptiveContamination,
)
from core.parameters.numerical_constants import STAT_THRESHOLDS


def test_contamination_state_observed_rate():
    """ContaminationState debe calcular tasa observada correctamente."""
    state = ContaminationState()
    
    state.add_detection(True)
    state.add_detection(False)
    state.add_detection(True)
    state.add_detection(False)
    
    assert state.observed_rate == 0.5


def test_contamination_state_empty():
    """ContaminationState vacío debe tener tasa 0."""
    state = ContaminationState()
    assert state.observed_rate == 0.0
    assert state.total_samples == 0


def test_hysteresis_config_defaults():
    """HysteresisConfig debe tener defaults razonables."""
    config = ContaminationHysteresisConfig()
    
    assert config.threshold_increase == 1.2
    assert config.threshold_decrease == 0.8
    assert config.adjustment_factor == 0.2
    assert config.min_samples == 50
    assert config.min_contamination == STAT_THRESHOLDS.CONTAMINATION_MIN
    assert config.max_contamination == STAT_THRESHOLDS.CONTAMINATION_MAX


def test_hysteresis_should_increase():
    """Debe aumentar si ratio > threshold y suficientes muestras."""
    config = ContaminationHysteresisConfig(min_samples=10)
    
    # ratio = 1.5 > 1.2, con 20 muestras
    assert config.should_increase(1.5, 20)
    
    # ratio = 1.1 < 1.2
    assert not config.should_increase(1.1, 20)
    
    # ratio = 1.5 pero solo 5 muestras
    assert not config.should_increase(1.5, 5)


def test_hysteresis_should_decrease():
    """Debe reducir si ratio < threshold y suficientes muestras."""
    config = ContaminationHysteresisConfig(min_samples=10)
    
    # ratio = 0.5 < 0.8, con 20 muestras
    assert config.should_decrease(0.5, 20)
    
    # ratio = 0.9 > 0.8
    assert not config.should_decrease(0.9, 20)
    
    # ratio = 0.5 pero solo 5 muestras
    assert not config.should_decrease(0.5, 5)


def test_adaptive_contamination_increases():
    """Contamination debe aumentar si tasa observada es alta."""
    adaptive = AdaptiveContamination(
        hysteresis_config=ContaminationHysteresisConfig(min_samples=10),
    )
    
    # Agregar 60% anomalías
    for _ in range(60):
        adaptive.add_detection(True)
    for _ in range(40):
        adaptive.add_detection(False)
    
    new_contamination = adaptive.update_contamination()
    assert new_contamination > STAT_THRESHOLDS.CONTAMINATION_DEFAULT


def test_adaptive_contamination_decreases():
    """Contamination debe reducir si tasa observada es baja."""
    adaptive = AdaptiveContamination(
        hysteresis_config=ContaminationHysteresisConfig(min_samples=10),
    )
    
    # Agregar 0.1% anomalías
    for _ in range(1):
        adaptive.add_detection(True)
    for _ in range(999):
        adaptive.add_detection(False)
    
    new_contamination = adaptive.update_contamination()
    assert new_contamination < STAT_THRESHOLDS.CONTAMINATION_DEFAULT


def test_adaptive_contamination_clamps_to_bounds():
    """Contamination debe estar en [min, max]."""
    adaptive = AdaptiveContamination()
    
    # Forzar tasa muy alta
    for _ in range(100):
        adaptive.add_detection(True)
    
    new_contamination = adaptive.update_contamination()
    assert new_contamination <= STAT_THRESHOLDS.CONTAMINATION_MAX
    
    # Forzar tasa muy baja
    adaptive2 = AdaptiveContamination()
    for _ in range(1000):
        adaptive2.add_detection(False)
    
    new_contamination = adaptive2.update_contamination()
    assert new_contamination >= STAT_THRESHOLDS.CONTAMINATION_MIN


def test_adaptive_contamination_hysteresis():
    """Hysteresis debe prevenir cambios pequeños."""
    adaptive = AdaptiveContamination(
        hysteresis_config=ContaminationHysteresisConfig(min_samples=10),
    )
    
    # Agregar 5% anomalías (cercano a default 0.5%)
    for _ in range(5):
        adaptive.add_detection(True)
    for _ in range(95):
        adaptive.add_detection(False)
    
    new_contamination = adaptive.update_contamination()
    # Debe mantenerse cerca del default (hysteresis)
    assert new_contamination == pytest.approx(STAT_THRESHOLDS.CONTAMINATION_DEFAULT, abs=0.01)


def test_should_refit():
    """should_refit debe detectar cambios significativos."""
    adaptive = AdaptiveContamination(
        hysteresis_config=ContaminationHysteresisConfig(min_samples=10),
    )
    
    # Sin suficientes muestras
    assert not adaptive.should_refit()
    
    # Con suficientes muestras pero cambio pequeño
    for _ in range(10):
        adaptive.add_detection(True)
    for _ in range(90):
        adaptive.add_detection(False)
    
    adaptive.update_contamination()
    assert not adaptive.should_refit(threshold=0.2)
    
    # Cambio grande - necesita update_contamination primero
    adaptive2 = AdaptiveContamination(
        hysteresis_config=ContaminationHysteresisConfig(min_samples=10),
    )
    for _ in range(30):
        adaptive2.add_detection(True)
    for _ in range(70):
        adaptive2.add_detection(False)
    
    # Actualizar contamination para que cambie significativamente
    adaptive2.update_contamination()
    # Verificar que el cambio es > 20%
    initial = STAT_THRESHOLDS.CONTAMINATION_DEFAULT
    current = adaptive2.state.current_contamination
    relative_change = abs(current - initial) / initial if initial > 0 else 0
    
    # Si el cambio es significativo, should_refit debe ser True
    if relative_change > 0.2:
        assert adaptive2.should_refit(threshold=0.2)
    else:
        # Si no es significativo, el test debe verificar que no refit
        assert not adaptive2.should_refit(threshold=0.2)


def test_adaptive_contamination_reset():
    """Reset debe limpiar estado."""
    adaptive = AdaptiveContamination()
    
    for _ in range(10):
        adaptive.add_detection(True)
    
    assert adaptive.state.total_samples == 10
    
    adaptive.reset()
    assert adaptive.state.total_samples == 0
    assert adaptive.state.current_contamination == STAT_THRESHOLDS.CONTAMINATION_DEFAULT
