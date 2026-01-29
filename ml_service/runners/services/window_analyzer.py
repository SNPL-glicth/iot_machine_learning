"""Window analyzer service for ML online processing.

Extraído de ml_stream_runner.py para modularidad.
Responsabilidad: Analizar ventanas deslizantes y clasificar patrones.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from iot_machine_learning.ml_service.config.ml_config import OnlineBehaviorConfig
from iot_machine_learning.ml_service.sliding_window_buffer import WindowStats
from ..models.online_analysis import OnlineAnalysis
from ..models.sensor_state import SensorState

logger = logging.getLogger(__name__)


class WindowAnalyzer:
    """Analiza ventanas deslizantes y clasifica patrones de comportamiento.
    
    Responsabilidades:
    - Calcular baseline, z-score, slopes
    - Detectar anomalías de curva
    - Detectar micro-variaciones
    - Clasificar patrón de comportamiento
    """
    
    def __init__(self, cfg: OnlineBehaviorConfig) -> None:
        self._cfg = cfg
    
    def analyze_windows(
        self,
        stats_by_window: Dict[str, WindowStats],
        prev_state: Optional[SensorState],
    ) -> OnlineAnalysis:
        """Analiza las ventanas y produce un OnlineAnalysis."""
        cfg = self._cfg

        w1 = stats_by_window.get("w1")
        w5 = stats_by_window.get("w5")
        w10 = stats_by_window.get("w10")

        baseline = w10 or w5 or w1 or next(iter(stats_by_window.values()))

        baseline_mean = baseline.mean
        baseline_std = baseline.std_dev
        last_value = (w1 or baseline).last_value

        z_score = 0.0
        if baseline_std > 0:
            z_score = (last_value - baseline_mean) / baseline_std

        slope_short = w1.trend if w1 else 0.0
        slope_medium = w5.trend if w5 else slope_short
        slope_long = w10.trend if w10 else slope_medium

        accel_short_vs_medium = slope_short - slope_medium
        accel_medium_vs_long = slope_medium - slope_long

        is_curve_anomalous = any(
            abs(s) >= cfg.slope_anomaly_threshold
            for s in (slope_short, slope_medium, slope_long)
        ) or any(
            abs(a) >= cfg.accel_anomaly_threshold
            for a in (accel_short_vs_medium, accel_medium_vs_long)
        )

        delta = last_value - baseline_mean
        has_microvariation = (
            abs(delta) >= cfg.microvariation_min_delta
            and abs(z_score) >= cfg.microvariation_z_score
            and not is_curve_anomalous
        )

        prev_in_transient = prev_state.in_transient_anomaly if prev_state else False
        outlier_for_transient = abs(z_score) >= cfg.transient_z_score

        new_transient = not prev_in_transient and outlier_for_transient
        recovered = prev_in_transient and not outlier_for_transient

        behavior_pattern = self._classify_pattern(
            w1=w1,
            w5=w5,
            w10=w10,
            baseline_mean=baseline_mean,
            z_score_last=z_score,
            is_curve_anomalous=is_curve_anomalous,
            has_microvariation=has_microvariation,
        )

        return OnlineAnalysis(
            behavior_pattern=behavior_pattern,
            is_curve_anomalous=is_curve_anomalous,
            has_microvariation=has_microvariation,
            microvariation_delta=delta,
            new_transient_anomaly=new_transient,
            recovered_transient=recovered,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            last_value=last_value,
            z_score_last=z_score,
            slope_short=slope_short,
            slope_medium=slope_medium,
            slope_long=slope_long,
            accel_short_vs_medium=accel_short_vs_medium,
            accel_medium_vs_long=accel_medium_vs_long,
        )

    def _classify_pattern(
        self,
        *,
        w1: Optional[WindowStats],
        w5: Optional[WindowStats],
        w10: Optional[WindowStats],
        baseline_mean: float,
        z_score_last: float,
        is_curve_anomalous: bool,
        has_microvariation: bool,
    ) -> str:
        """Clasificación cualitativa del patrón de comportamiento del sensor."""

        ref = w10 or w5 or w1
        if ref is None:
            return "STABLE"

        var_long = ref.std_dev
        trend_long = ref.trend

        # Heurísticas simples por patrón
        if abs(trend_long) < 0.05 and var_long < 0.01 and abs(z_score_last) < 1.0:
            return "STABLE"

        # Oscilación: alta variabilidad y cambios de signo en la pendiente
        if var_long >= 0.05:
            t1 = w1.trend if w1 else trend_long
            t5 = w5.trend if w5 else trend_long
            if (t1 > 0 and t5 < 0) or (t1 < 0 and t5 > 0):
                return "OSCILLATING"

        # Deriva: tendencia sostenida sin alta variabilidad
        if abs(trend_long) >= 0.05 and var_long < 0.05:
            return "DRIFTING"

        # Spike: z_score muy alto
        if abs(z_score_last) >= 3.0:
            return "SPIKE"

        # Micro-variación
        if has_microvariation:
            return "MICRO_VARIATION"

        # Curva anómala
        if is_curve_anomalous:
            return "CURVE_ANOMALY"

        return "STABLE"
