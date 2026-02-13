"""Análisis estructural unificado de una serie temporal — Nivel 1.

Consolida en un único value object las propiedades estructurales que
antes se calculaban de forma dispersa en:
- Taylor engine (slope, curvature, stability, accel_variance)
- Pattern detector (noise_ratio, regime classification)
- Series profile (stationarity, volatility)

Consumido por: prediction engines, anomaly detection, pattern detection,
cognitive layer.

Value object puro — sin I/O, sin estado, sin dependencias de infraestructura.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class RegimeType(Enum):
    """Régimen dinámico de la serie."""

    STABLE = "stable"
    TRENDING = "trending"
    VOLATILE = "volatile"
    NOISY = "noisy"


@dataclass(frozen=True)
class StructuralAnalysis:
    """Análisis estructural unificado de una serie temporal.

    Combina propiedades de primer y segundo orden con clasificación
    de régimen y métricas de calidad de señal.

    Attributes:
        slope: Pendiente local (f'(t)) — tasa de cambio instantánea.
        curvature: Curvatura local (f''(t)) — aceleración.
        stability: Indicador de estabilidad (0.0=estable, 1.0=inestable).
        accel_variance: Varianza de la aceleración a lo largo de la ventana.
        noise_ratio: Ratio σ/|μ| — proporción de ruido vs señal.
        regime: Clasificación del régimen dinámico.
        mean: Media de la serie.
        std: Desviación estándar de la serie.
        trend_strength: Fuerza de la tendencia (|slope| / max(|mean|, ε)).
        n_points: Número de puntos analizados.
    """

    slope: float = 0.0
    curvature: float = 0.0
    stability: float = 0.0
    accel_variance: float = 0.0
    noise_ratio: float = 0.0
    regime: RegimeType = RegimeType.STABLE
    mean: float = 0.0
    std: float = 0.0
    trend_strength: float = 0.0
    n_points: int = 0
    dt: float = 1.0

    @property
    def is_stable(self) -> bool:
        """True si el régimen es estable."""
        return self.regime == RegimeType.STABLE

    @property
    def is_trending(self) -> bool:
        """True si hay tendencia significativa."""
        return self.regime == RegimeType.TRENDING

    @property
    def is_noisy(self) -> bool:
        """True si la señal es predominantemente ruido."""
        return self.regime == RegimeType.NOISY

    @property
    def has_sufficient_data(self) -> bool:
        """True si hay suficientes puntos para análisis confiable (≥5)."""
        return self.n_points >= 5

    def to_dict(self) -> dict:
        """Serializa para audit logging / metadata."""
        return {
            "slope": round(self.slope, 8),
            "curvature": round(self.curvature, 8),
            "stability": round(self.stability, 6),
            "accel_variance": round(self.accel_variance, 8),
            "noise_ratio": round(self.noise_ratio, 6),
            "regime": self.regime.value,
            "mean": round(self.mean, 8),
            "std": round(self.std, 8),
            "trend_strength": round(self.trend_strength, 6),
            "n_points": self.n_points,
            "dt": self.dt,
        }

    def to_feature_vector(self) -> List[float]:
        """Vector de features para modelos ML.

        Orden: [slope, curvature, stability, accel_variance,
                noise_ratio, trend_strength]
        """
        return [
            self.slope,
            self.curvature,
            self.stability,
            self.accel_variance,
            self.noise_ratio,
            self.trend_strength,
        ]

    @classmethod
    def empty(cls) -> StructuralAnalysis:
        """Factory para serie vacía o insuficiente."""
        return cls()

    @classmethod
    def from_taylor_diagnostic(
        cls,
        diagnostic: object,
        values: Optional[List[float]] = None,
    ) -> StructuralAnalysis:
        """Construye desde un TaylorDiagnostic existente.

        Bridge: permite reusar diagnósticos Taylor ya calculados
        sin recomputar desde cero.

        Args:
            diagnostic: ``TaylorDiagnostic`` (typed as object to avoid
                importing infrastructure types in domain).
            values: Valores originales (para calcular mean/std/noise_ratio).

        Returns:
            ``StructuralAnalysis`` con datos del diagnóstico Taylor.
        """
        slope = getattr(diagnostic, "local_slope", 0.0)
        curvature = getattr(diagnostic, "curvature", 0.0)
        stability = getattr(diagnostic, "stability_indicator", 0.0)
        accel_var = getattr(diagnostic, "accel_variance", 0.0)

        mean = 0.0
        std = 0.0
        noise_ratio = 0.0
        n = 0

        if values:
            n = len(values)
            if n > 0:
                mean = sum(values) / n
                variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
                std = math.sqrt(variance)
                noise_ratio = std / abs(mean) if abs(mean) > 1e-9 else 0.0

        mean_ref = abs(mean) if abs(mean) > 1e-9 else 1.0
        trend_strength = abs(slope) / mean_ref

        regime = _classify_regime(noise_ratio, slope, std, mean)

        return cls(
            slope=slope,
            curvature=curvature,
            stability=stability,
            accel_variance=accel_var,
            noise_ratio=noise_ratio,
            regime=regime,
            mean=mean,
            std=std,
            trend_strength=trend_strength,
            n_points=n,
        )


def _classify_regime(
    noise_ratio: float,
    slope: float,
    std: float,
    mean: float = 0.0,
) -> RegimeType:
    """Clasificación de régimen basada en umbrales.

    Categories:
        stable   — low noise, low slope
        trending — significant slope relative to signal magnitude
        noisy    — high noise ratio (σ/|μ| > 0.5)
        volatile — moderate noise ratio (0.15 < σ/|μ| ≤ 0.5)
    """
    if noise_ratio > 0.5:
        return RegimeType.NOISY
    abs_slope = abs(slope)
    mean_ref = abs(mean) if abs(mean) > 1e-9 else 1.0
    slope_ratio = abs_slope / mean_ref
    if slope_ratio > 0.005 and abs_slope > 0.01:
        return RegimeType.TRENDING
    if std > 0 and noise_ratio > 0.15:
        return RegimeType.VOLATILE
    return RegimeType.STABLE
