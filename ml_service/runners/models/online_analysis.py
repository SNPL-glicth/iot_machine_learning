"""Online analysis model for ML processing."""

from dataclasses import dataclass


@dataclass
class OnlineAnalysis:
    """Resultado del análisis de ventanas deslizantes.
    
    Contiene todas las métricas calculadas del análisis online:
    - Patrón de comportamiento
    - Indicadores de anomalía
    - Estadísticas de baseline
    - Métricas de tendencia
    """
    
    behavior_pattern: str
    is_curve_anomalous: bool
    has_microvariation: bool
    microvariation_delta: float
    new_transient_anomaly: bool
    recovered_transient: bool
    baseline_mean: float
    baseline_std: float
    last_value: float
    z_score_last: float
    slope_short: float
    slope_medium: float
    slope_long: float
    accel_short_vs_medium: float
    accel_medium_vs_long: float
