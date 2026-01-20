from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
    """Configuración de Isolation Forest por sensor."""

    contamination: float = 0.05
    n_estimators: int = 100
    random_state: int = 42


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
class GlobalMLConfig:
    regression: RegressionConfig = RegressionConfig()
    anomaly: AnomalyConfig = AnomalyConfig()
    online: OnlineBehaviorConfig = OnlineBehaviorConfig()


# Config global por defecto utilizable en runners/servicios
DEFAULT_ML_CONFIG = GlobalMLConfig()
