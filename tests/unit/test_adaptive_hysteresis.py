"""Tests para adaptive scaling con hysteresis."""
import pytest

from core.drift.adaptive_strategy import (
    AdaptiveScaler,
    HysteresisConfig,
    UnifiedAdaptiveConfig,
)
from core.drift.drift_coupling import (
    AdaptiveScalerDriftListener,
    DriftEvent,
    DriftNotifier,
)


def test_hysteresis_prevents_oscillation():
    """Hysteresis debe prevenir cambios rápidos."""
    scaler = AdaptiveScaler(
        scale_min=0.5,
        scale_max=5.0,
        hysteresis_config=HysteresisConfig(
            threshold_increase=1.2,
            threshold_decrease=0.8,
            min_samples=5,
        ),
    )
    
    base_vol = 1.0
    
    # Primera pasada: ratio = 1.1 (dentro de banda [0.8, 1.2])
    scale1 = scaler.compute_scale(current_volatility=1.1, base_volatility=base_vol)
    assert scale1 == pytest.approx(1.0, abs=0.1)
    
    # Segunda pasada: ratio = 0.9 (dentro de banda)
    scale2 = scaler.compute_scale(current_volatility=0.9, base_volatility=base_vol)
    assert scale2 == pytest.approx(1.0, abs=0.1)
    
    # Tercera pasada: ratio = 1.5 (fuera de banda)
    # Con smoothing, el ratio se mantiene cerca de 1.0 inicialmente
    # La hysteresis previene cambios rápidos
    for _ in range(5):
        scaler.compute_scale(current_volatility=1.5, base_volatility=base_vol)
    
    scale3 = scaler.state.current_scale
    # Scale debe mantenerse cerca de 1.0 debido a smoothing + hysteresis
    assert scale3 == pytest.approx(1.0, abs=0.2)
    
    # Cuarta pasada: ratio = 3.0 (muy fuera de banda)
    for _ in range(20):
        scaler.compute_scale(current_volatility=3.0, base_volatility=base_vol)
    
    scale4 = scaler.state.current_scale
    # Después de muchas iteraciones, scale debe aumentar gradualmente
    assert scale4 > 1.2


def test_drift_resets_scaler():
    """Drift debe resetear el estado del scaler."""
    scaler = AdaptiveScaler()
    listener = AdaptiveScalerDriftListener(scaler)
    notifier = DriftNotifier()
    notifier.subscribe(listener)
    
    # Agregar historia al scaler
    for i in range(10):
        scaler.compute_scale(current_volatility=2.0, base_volatility=1.0)
    
    assert len(scaler.state.volatility_history) > 0
    
    # Notificar drift
    event = DriftEvent.create_now(magnitude=0.5, detector="test")
    notifier.notify(event)
    
    # Estado debe resetearse
    assert len(scaler.state.volatility_history) == 0
    assert scaler.state.current_scale == 1.0


def test_gradual_transition_limits_change():
    """Transición gradual debe limitar cambios bruscos."""
    scaler = AdaptiveScaler(
        scale_min=0.5,
        scale_max=5.0,
        hysteresis_config=HysteresisConfig(min_samples=1),
    )
    
    # Volatilidad salta de 1.0 a 10.0
    scaler.compute_scale(current_volatility=1.0, base_volatility=1.0)
    scaler.compute_scale(current_volatility=10.0, base_volatility=1.0)
    
    # El cambio debe ser limitado a max_change=0.5 por step
    # Scale no debería saltar de 1.0 a 10.0 en un solo step
    assert scaler.state.current_scale < 2.0


def test_hysteresis_config_defaults():
    """Configuración de hysteresis debe tener defaults razonables."""
    config = HysteresisConfig()
    
    assert config.threshold_increase == 1.2
    assert config.threshold_decrease == 0.8
    assert config.smooth_factor == 0.3
    assert config.min_samples == 5


def test_unified_adaptive_config():
    """Configuración unificada debe ser consistente."""
    assert UnifiedAdaptiveConfig.ADAPTIVE_ENABLED is True
    assert UnifiedAdaptiveConfig.SCALE_MIN == 0.5
    assert UnifiedAdaptiveConfig.SCALE_MAX == 5.0
    assert UnifiedAdaptiveConfig.HYSTERESIS_INCREASE == 1.2
    assert UnifiedAdaptiveConfig.HYSTERESIS_DECREASE == 0.8
    assert UnifiedAdaptiveConfig.SMOOTH_FACTOR == 0.3


def test_adaptive_state_mean_volatility():
    """AdaptiveState debe calcular volatilidad promedio correctamente."""
    from core.drift.adaptive_strategy import AdaptiveState
    
    state = AdaptiveState()
    state.add_volatility(1.0)
    state.add_volatility(2.0)
    state.add_volatility(3.0)
    
    assert state.mean_volatility() == pytest.approx(2.0)


def test_scaler_clamps_to_bounds():
    """Scaler debe clamp scale a [scale_min, scale_max]."""
    scaler = AdaptiveScaler(
        scale_min=0.5,
        scale_max=2.0,
        hysteresis_config=HysteresisConfig(min_samples=1),
    )
    
    # Volatilidad muy alta → scale debería estar limitado a scale_max
    for _ in range(10):
        scaler.compute_scale(current_volatility=100.0, base_volatility=1.0)
    
    assert scaler.state.current_scale <= 2.0


def test_drift_notifier_singleton():
    """DriftNotifier debe ser singleton."""
    notifier1 = DriftNotifier()
    notifier2 = DriftNotifier()
    
    assert notifier1 is notifier2


def test_drift_event_creation():
    """DriftEvent.create_now debe crear evento con timestamp actual."""
    event = DriftEvent.create_now(magnitude=0.5, detector="test")
    
    assert event.magnitude == 0.5
    assert event.detector_name == "test"
    assert event.timestamp > 0
    import time
    assert time.time() - event.timestamp < 1.0  # Dentro de 1 segundo
