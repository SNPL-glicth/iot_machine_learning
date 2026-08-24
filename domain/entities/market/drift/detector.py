"""FASE 10.4 — Drift Detection: vigilancia continua de degradación temporal.

Objetivo:
- Detectar drift/degradación en performance temporal
- Comparar ventanas: Last 100, 500, 1,000, 5,000, Lifetime
- Conectar con sistema Page-Hinkley ya existente
- Alertar cuando la performance reciente es significativamente peor
- Diferenciar entre ruido normal y drift real

Sistema de detección:
1. Window comparison: comparar ventanas temporales con tests estadísticos
2. Page-Hinkley: detección de cambio de media acumulativo
3. CUSUM: detección de cambio en media
4. ADWIN: adaptive windowing para drift conceptual

Cuando se detecta drift:
- STATUS = POSSIBLE DRIFT
- Registrar cuándo y dónde se detectó
- Sugerir acciones (reentrenar, cambiar estrategia, etc.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final
from collections import deque

__all__ = [
    "DriftStatus",
    "DriftConfig",
    "WindowMetrics",
    "DriftAlert",
    "DriftDetector",
    "page_hinkley_test",
]


class DriftStatus(Enum):
    """Estado de detección de drift."""
    
    STABLE = "stable"
    POSSIBLE_DRIFT = "possible_drift"
    CONFIRMED_DRIFT = "confirmed_drift"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class DriftConfig:
    """Configuración del detector de drift."""
    
    # Ventanas temporales
    windows: tuple[int, ...] = (100, 500, 1000, 5000)  # Last N samples
    
    # Umbrales de detección
    min_samples: int = 100  # Mínimo de muestras para evaluar
    degradation_threshold: float = 0.05  # 5% de degradación significativa
    confidence_threshold: float = 0.95  # 95% confianza para drift
    
    # Page-Hinkley
    ph_alpha: float = 0.05  # Significance level
    ph_min_samples: int = 30  # Mínimo para PH test
    ph_lambda: float | None = None  # Threshold (si None, se calcula dinámicamente)
    
    def __post_init__(self) -> None:
        if self.min_samples < 1:
            raise ValueError(f"min_samples debe ser >= 1: {self.min_samples}")
        if not 0.0 < self.degradation_threshold < 1.0:
            raise ValueError(f"degradation_threshold inválido: {self.degradation_threshold}")


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    """Métricas de una ventana temporal."""
    
    window_size: int
    accuracy: float
    mean_return: float
    std_return: float
    n: int
    
    @property
    def sharpe(self) -> float:
        """Sharpe ratio simplificado."""
        if self.std_return == 0:
            return 0.0
        return self.mean_return / self.std_return


@dataclass(frozen=True, slots=True)
class DriftAlert:
    """Alerta de drift detectado."""
    
    status: DriftStatus
    window: str  # "last_100", "last_500", etc.
    current_accuracy: float
    baseline_accuracy: float
    degradation: float
    p_value: float | None
    confidence: float
    detected_at: int  # Sample number cuando se detectó
    reason: str
    
    @property
    def is_significant(self) -> bool:
        """True si el drift es estadísticamente significativo."""
        return self.confidence >= 0.95


def page_hinkley_test(
    values: list[float],
    alpha: float = 0.05,
    min_samples: int = 30,
    lambda_threshold: float | None = None,
) -> tuple[bool, float, float]:
    """Page-Hinkley test para detección de cambio de media.
    
    Returns:
        (drift_detected, statistic, threshold)
    """
    if len(values) < min_samples:
        return False, 0.0, 0.0
    
    # Calcular media acumulativa y mínimos/máximos
    cumulative_mean = []
    running_sum = 0.0
    for i, x in enumerate(values, 1):
        running_sum += x
        cumulative_mean.append(running_sum / i)
    
    # Statistic: max de (media_acumulada - min_media_acumulada)
    min_mean = min(cumulative_mean)
    ph_statistic = max(m - min_mean for m in cumulative_mean)
    
    # Calcular threshold si no se proporciona
    if lambda_threshold is None:
        # Approx: lambda = sqrt(2*n*ln(1/alpha)) * sigma
        mean_val = sum(values) / len(values)
        sigma = math.sqrt(sum((x - mean_val)**2 for x in values) / (len(values) - 1)) if len(values) > 1 else 1.0
        n = len(values)
        lambda_threshold = math.sqrt(2 * n * math.log(1/alpha)) * sigma
    
    drift_detected = ph_statistic > lambda_threshold
    return drift_detected, ph_statistic, lambda_threshold


class DriftDetector:
    """Detector de drift con múltiples métodos."""
    
    def __init__(self, config: DriftConfig | None = None) -> None:
        self.config = config or DriftConfig()
        self._values: deque[float] = deque(maxlen=max(self.config.windows) * 2)
        self._outcomes: deque[bool] = deque(maxlen=max(self.config.windows) * 2)
        self._sample_count = 0
        self._baseline_accuracy: float | None = None
        self._alerts: list[DriftAlert] = []
    
    def add_observation(self, value: float, outcome: bool | None = None) -> None:
        """Agrega una observación al detector."""
        self._values.append(value)
        if outcome is not None:
            self._outcomes.append(outcome)
        self._sample_count += 1
        
        # Establecer baseline después de min_samples
        if self._baseline_accuracy is None and len(self._outcomes) >= self.config.min_samples:
            self._baseline_accuracy = sum(1 for o in self._outcomes if o) / len(self._outcomes)
    
    def _get_window_metrics(self, window_size: int) -> WindowMetrics | None:
        """Calcula métricas para una ventana temporal."""
        if len(self._values) < window_size:
            return None
        
        recent_values = list(self._values)[-window_size:]
        recent_outcomes = list(self._outcomes)[-window_size:] if self._outcomes else []
        
        accuracy = sum(1 for o in recent_outcomes if o) / len(recent_outcomes) if recent_outcomes else 0.0
        mean_return = sum(recent_values) / len(recent_values)
        std_return = math.sqrt(sum((x - mean_return) ** 2 for x in recent_values) / (len(recent_values) - 1)) if len(recent_values) > 1 else 0.0
        
        return WindowMetrics(
            window_size=window_size,
            accuracy=accuracy,
            mean_return=mean_return,
            std_return=std_return,
            n=len(recent_values),
        )
    
    def _compare_windows(self, current: WindowMetrics, baseline: WindowMetrics) -> tuple[bool, float]:
        """Compara dos ventanas con test estadístico.
        
        Returns (significant_drift, p_value_approx)
        """
        if current.n < self.config.min_samples or baseline.n < self.config.min_samples:
            return False, 1.0
        
        # Degradación en accuracy
        degradation = baseline.accuracy - current.accuracy
        
        if degradation < self.config.degradation_threshold:
            return False, 1.0
        
        # Test de proporciones (aproximación normal)
        p1 = baseline.accuracy
        p2 = current.accuracy
        n1 = baseline.n
        n2 = current.n
        
        pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
        
        if se == 0:
            return False, 1.0
        
        z_score = (p2 - p1) / se
        # p-value aproximado (two-tailed)
        if z_score >= 0:
            p_value = 2 * (1 - (0.5 * (1 + math.erf(z_score / math.sqrt(2)))))
        else:
            p_value = 2 * (0.5 * (1 + math.erf(z_score / math.sqrt(2))))
        
        is_significant = p_value < (1 - self.config.confidence_threshold)
        return is_significant, p_value
    
    def detect_drift(self) -> DriftAlert | None:
        """Detecta drift en las métricas actuales."""
        if len(self._values) < self.config.min_samples:
            return None
        
        if self._baseline_accuracy is None:
            return None
        
        # Evaluar cada ventana
        for window_size in self.config.windows:
            current_metrics = self._get_window_metrics(window_size)
            if current_metrics is None:
                continue
            
            # Crear baseline metrics
            baseline_metrics = WindowMetrics(
                window_size=window_size,
                accuracy=self._baseline_accuracy,
                mean_return=sum(self._values) / len(self._values),
                std_return=math.sqrt(sum((x - sum(self._values)/len(self._values))**2 for x in self._values) / (len(self._values) - 1)) if len(self._values) > 1 else 0.0,
                n=len(self._values),
            )
            
            # Comparar ventanas
            is_significant, p_value = self._compare_windows(current_metrics, baseline_metrics)
            
            degradation = baseline_metrics.accuracy - current_metrics.accuracy
            
            if is_significant:
                # Drift detectado
                alert = DriftAlert(
                    status=DriftStatus.CONFIRMED_DRIFT if degradation > 0.10 else DriftStatus.POSSIBLE_DRIFT,
                    window=f"last_{window_size}",
                    current_accuracy=current_metrics.accuracy,
                    baseline_accuracy=baseline_metrics.accuracy,
                    degradation=degradation,
                    p_value=p_value,
                    confidence=self.config.confidence_threshold,
                    detected_at=self._sample_count,
                    reason=f"Accuracy en last_{window_size} ({current_metrics.accuracy:.2%}) vs baseline ({baseline_metrics.accuracy:.2%})",
                )
                self._alerts.append(alert)
                return alert
        
        # También ejecutar Page-Hinkley
        if len(self._values) >= self.config.ph_min_samples:
            # Convertir outcomes a valores (1=success, 0=failure)
            ph_values = [1.0 if o else 0.0 for o in self._outcomes] if self._outcomes else []
            if ph_values:
                drift_detected, statistic, threshold = page_hinkley_test(
                    ph_values,
                    alpha=self.config.ph_alpha,
                    min_samples=self.config.ph_min_samples,
                    lambda_threshold=self.config.ph_lambda,
                )
                
                if drift_detected:
                    alert = DriftAlert(
                        status=DriftStatus.CONFIRMED_DRIFT,
                        window="page_hinkley",
                        current_accuracy=sum(1 for o in self._outcomes if o) / len(self._outcomes),
                        baseline_accuracy=self._baseline_accuracy,
                        degradation=self._baseline_accuracy - (sum(1 for o in self._outcomes if o) / len(self._outcomes)),
                        p_value=self.config.ph_alpha,
                        confidence=self.config.confidence_threshold,
                        detected_at=self._sample_count,
                        reason=f"Page-Hinkley statistic {statistic:.4f} > threshold {threshold:.4f}",
                    )
                    self._alerts.append(alert)
                    return alert
        
        return None
    
    def get_all_window_metrics(self) -> dict[str, WindowMetrics]:
        """Obtiene métricas de todas las ventanas."""
        metrics = {}
        
        # Lifetime
        if len(self._values) >= self.config.min_samples:
            metrics["lifetime"] = WindowMetrics(
                window_size=len(self._values),
                accuracy=sum(1 for o in self._outcomes if o) / len(self._outcomes) if self._outcomes else 0.0,
                mean_return=sum(self._values) / len(self._values),
                std_return=math.sqrt(sum((x - sum(self._values)/len(self._values))**2 for x in self._values) / (len(self._values) - 1)) if len(self._values) > 1 else 0.0,
                n=len(self._values),
            )
        
        # Windows configuradas
        for window_size in self.config.windows:
            window_metrics = self._get_window_metrics(window_size)
            if window_metrics:
                metrics[f"last_{window_size}"] = window_metrics
        
        return metrics
    
    def get_alerts(self) -> list[DriftAlert]:
        """Obtiene todas las alertas de drift."""
        return self._alerts.copy()
    
    def reset(self) -> None:
        """Resetea el detector."""
        self._values.clear()
        self._outcomes.clear()
        self._sample_count = 0
        self._baseline_accuracy = None
        self._alerts.clear()


def render_drift_report(detector: DriftDetector) -> str:
    """Renderiza reporte de detección de drift."""
    lines = [
        "DRIFT DETECTION REPORT",
        "=" * 24,
        "",
        f"Sample count: {detector._sample_count}",
        f"Baseline accuracy: {detector._baseline_accuracy:.2%}" if detector._baseline_accuracy else "Baseline accuracy: not established",
        "",
        "WINDOW METRICS",
        "-" * 15,
    ]
    
    metrics = detector.get_all_window_metrics()
    for window_name, window_metrics in sorted(metrics.items()):
        lines.append(f"{window_name:<12} accuracy={window_metrics.accuracy:.2%} sharpe={window_metrics.sharpe:.3f} n={window_metrics.n}")
    
    # Alertas
    alerts = detector.get_alerts()
    if alerts:
        lines.append("")
        lines.append("DRIFT ALERTS")
        lines.append("-" * 13)
        for alert in alerts:
            marker = "🚨" if alert.status == DriftStatus.CONFIRMED_DRIFT else "⚠️"
            lines.append(f"{marker} {alert.window}: {alert.reason}")
            lines.append(f"   Current: {alert.current_accuracy:.2%} vs Baseline: {alert.baseline_accuracy:.2%}")
            lines.append(f"   Degradation: {alert.degradation:.2%}, Confidence: {alert.confidence:.0%}")
    else:
        lines.append("")
        lines.append("STATUS: STABLE (no drift detected)")
    
    return "\n".join(lines)