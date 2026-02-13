"""Features temporales derivadas de una serie — Nivel 1 (Matemático).

Representa las propiedades dinámicas de una serie temporal:
velocidad (dv/dt), aceleración (d²v/dt²), jitter (variabilidad de Δt),
y estadísticas agregadas sobre estas magnitudes.

Value object puro — sin I/O, sin estado, sin dominio.
Consumido por anomaly detection, pattern detection y cognitive layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class TemporalFeatures:
    """Features dinámicas derivadas de una serie temporal.

    Todas las magnitudes son calculadas respecto al tiempo real (Δt),
    no por conteo de muestras.

    Attributes:
        velocities: dv/dt para cada par consecutivo de puntos.
        accelerations: d²v/dt² para cada tripleta consecutiva.
        dt_values: Intervalos Δt entre puntos consecutivos.
        mean_velocity: Media de |dv/dt|.
        std_velocity: Desviación estándar de dv/dt.
        max_abs_velocity: Máximo |dv/dt| observado.
        mean_acceleration: Media de |d²v/dt²|.
        std_acceleration: Desviación estándar de d²v/dt².
        max_abs_acceleration: Máximo |d²v/dt²| observado.
        jitter: Coeficiente de variación de Δt (σ_dt / μ_dt).
            0.0 = muestreo perfectamente uniforme.
        mean_dt: Intervalo medio entre puntos.
        last_velocity: Velocidad instantánea más reciente.
        last_acceleration: Aceleración instantánea más reciente.
        n_points: Número de puntos en la serie original.
    """

    velocities: List[float] = field(default_factory=list)
    accelerations: List[float] = field(default_factory=list)
    dt_values: List[float] = field(default_factory=list)
    mean_velocity: float = 0.0
    std_velocity: float = 0.0
    max_abs_velocity: float = 0.0
    mean_acceleration: float = 0.0
    std_acceleration: float = 0.0
    max_abs_acceleration: float = 0.0
    jitter: float = 0.0
    mean_dt: float = 1.0
    last_velocity: float = 0.0
    last_acceleration: float = 0.0
    n_points: int = 0

    @property
    def has_velocity(self) -> bool:
        """True si hay al menos una velocidad calculada (≥2 puntos)."""
        return len(self.velocities) > 0

    @property
    def has_acceleration(self) -> bool:
        """True si hay al menos una aceleración calculada (≥3 puntos)."""
        return len(self.accelerations) > 0

    @property
    def is_uniform_sampling(self) -> bool:
        """True si el muestreo es aproximadamente uniforme (jitter < 0.1)."""
        return self.jitter < 0.1

    def to_feature_vector(self) -> List[float]:
        """Retorna vector de features para uso en modelos ML.

        Orden: [mean_velocity, std_velocity, max_abs_velocity,
                mean_acceleration, std_acceleration, max_abs_acceleration,
                jitter, last_velocity, last_acceleration]
        """
        return [
            self.mean_velocity,
            self.std_velocity,
            self.max_abs_velocity,
            self.mean_acceleration,
            self.std_acceleration,
            self.max_abs_acceleration,
            self.jitter,
            self.last_velocity,
            self.last_acceleration,
        ]

    def to_dict(self) -> dict:
        """Serializa para audit logging / metadata."""
        return {
            "mean_velocity": round(self.mean_velocity, 8),
            "std_velocity": round(self.std_velocity, 8),
            "max_abs_velocity": round(self.max_abs_velocity, 8),
            "mean_acceleration": round(self.mean_acceleration, 8),
            "std_acceleration": round(self.std_acceleration, 8),
            "max_abs_acceleration": round(self.max_abs_acceleration, 8),
            "jitter": round(self.jitter, 6),
            "mean_dt": round(self.mean_dt, 6),
            "last_velocity": round(self.last_velocity, 8),
            "last_acceleration": round(self.last_acceleration, 8),
            "n_points": self.n_points,
            "n_velocities": len(self.velocities),
            "n_accelerations": len(self.accelerations),
        }

    @classmethod
    def empty(cls) -> TemporalFeatures:
        """Factory para serie vacía o insuficiente."""
        return cls()
