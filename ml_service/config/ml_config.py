from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal

#tiene que haber una configuracion de regresion 
@dataclass(frozen=True)
class RegressionConfig:
    """Configuración de regresión por sensor (Linear / Ridge)."""

    model_type: Literal["linear", "ridge"] = "ridge"
    window_points: int = 500
    min_points: int = 20
    horizon_minutes: int = 10
    min_confidence: float = 0.2
    max_confidence: float = 0.95


@dataclass(frozen=True)
class AnomalyConfig:
    """Configuración de detección de anomalías (IF, LOF, Z-score, IQR)."""

    contamination: float = 0.05
    n_estimators: int = 100
    random_state: int = 42

    # LOF
    lof_max_neighbors: int = 20

    # Minimum training points for VotingAnomalyDetector
    min_training_points: int = 50

    # Z-score vote thresholds (z > upper → 1.0, lower < z <= upper → linear)
    z_vote_lower: float = 2.0
    z_vote_upper: float = 3.0

    # Severity score cutoffs: [none, low, medium, high, critical]
    severity_none_max: float = 0.3
    severity_low_max: float = 0.5
    severity_medium_max: float = 0.7
    severity_high_max: float = 0.9


@dataclass(frozen=True)
class OnlineBehaviorConfig:
    """Configuración de ML online orientado a comportamiento.

    Estos valores son heurísticos y se pueden ajustar según dominio.
    """

    # Nº mínimo de puntos por ventana para considerarla estable
    min_points_per_window: int = 3

    # Umbral de pendiente (valor/segundo) para considerar un cambio fuerte
    slope_anomaly_threshold: float = 0.5

    # Umbral de diferencia de pendiente entre ventanas (aceleración aproximada)
    accel_anomaly_threshold: float = 0.5

    # Z-score por encima del cual una microvariación se considera relevante
    microvariation_z_score: float = 3.0

    # Cambio mínimo absoluto para microvariaciones (para ignorar ruido numérico)
    microvariation_min_delta: float = 0.01

    # Z-score para marcar anomalías transitorias (salida de rango clara)
    transient_z_score: float = 2.5

    # Errores de predicción
    prediction_error_relative: float = 0.2  # 20% por defecto
    prediction_error_absolute: float = 0.5  # valor absoluto mínimo
    prediction_time_tolerance_seconds: int = 60
    dedupe_minutes_prediction_deviation: int = 5


@dataclass(frozen=True)
class EngineConfig:
    """Configuración de motores de predicción UTSAE (Fase 1).

    Estos valores se usan como defaults cuando los feature flags no
    especifican overrides.  Los feature flags tienen prioridad.
    """

    # Motor por defecto: "baseline_moving_average" | "taylor"
    default_engine: str = "baseline_moving_average"

    # Taylor
    taylor_order: int = 2
    taylor_horizon: int = 1
    taylor_trend_threshold: float = 0.01
    taylor_min_confidence: float = 0.3
    taylor_max_confidence: float = 0.95

    # Kalman
    kalman_Q: float = 1e-5
    kalman_warmup_size: int = 10

    # Clamp
    clamp_margin_pct: float = 0.3

    # Per-sensor overrides: {sensor_id: engine_name}
    sensor_overrides: Dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GlobalMLConfig:
    regression: RegressionConfig = RegressionConfig()
    anomaly: AnomalyConfig = AnomalyConfig()
    online: OnlineBehaviorConfig = OnlineBehaviorConfig()
    engine: EngineConfig = EngineConfig()


# Config global por defecto utilizable en runners/servicios
DEFAULT_ML_CONFIG = GlobalMLConfig()
