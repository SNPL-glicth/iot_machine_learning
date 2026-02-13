"""Snapshots de señal y filtrado — tipos de integración.

Capturan el estado de la señal y del filtrado en el momento de la
inferencia.  Son domain-pure: no importan nada de infraestructura.
Reciben datos ya computados y los encapsulan como value objects
inmutables y serializables.

Responsabilidad:
- ``SignalSnapshot``: perfil estructural de la señal de entrada.
- ``FilterSnapshot``: resultado del filtrado aplicado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class SignalSnapshot:
    """Perfil estructural de la señal en el momento de inferencia.

    Mapea 1:1 con los campos de ``StructuralAnalysis`` y ``SignalProfile``
    pero sin depender de ellos.  El orquestador los copia al construir
    la ``Explanation``.

    Attributes:
        n_points: Puntos en la ventana.
        mean: Media aritmética.
        std: Desviación estándar.
        noise_ratio: σ / |μ| (coeficiente de variación).
        slope: Pendiente lineal estimada.
        curvature: Curvatura (segunda derivada).
        regime: Régimen detectado (``"stable"``, ``"trending"``, etc.).
        dt: Paso temporal estimado.
        extra: Campos adicionales (extensibilidad).
    """

    n_points: int = 0
    mean: float = 0.0
    std: float = 0.0
    noise_ratio: float = 0.0
    slope: float = 0.0
    curvature: float = 0.0
    regime: str = "unknown"
    dt: float = 1.0
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "n_points": self.n_points,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "noise_ratio": round(self.noise_ratio, 6),
            "slope": round(self.slope, 6),
            "curvature": round(self.curvature, 6),
            "regime": self.regime,
            "dt": self.dt,
        }
        if self.extra:
            d["extra"] = dict(self.extra)
        return d

    @classmethod
    def empty(cls) -> SignalSnapshot:
        return cls()

    @classmethod
    def from_dict(cls, data: dict) -> SignalSnapshot:
        """Reconstruye desde un dict serializado."""
        return cls(
            n_points=data.get("n_points", 0),
            mean=data.get("mean", 0.0),
            std=data.get("std", 0.0),
            noise_ratio=data.get("noise_ratio", 0.0),
            slope=data.get("slope", 0.0),
            curvature=data.get("curvature", 0.0),
            regime=data.get("regime", "unknown"),
            dt=data.get("dt", 1.0),
            extra=data.get("extra", {}),
        )


@dataclass(frozen=True)
class FilterSnapshot:
    """Resultado del filtrado aplicado a la señal.

    Mapea 1:1 con ``FilterDiagnostic`` pero sin depender de él.

    Attributes:
        filter_name: Nombre del filtro (``"KalmanSignalFilter"``, etc.).
        n_points: Puntos procesados.
        noise_reduction_ratio: 1 - (filtered_std / raw_std).
        mean_absolute_error: MAE entre crudo y filtrado.
        max_absolute_error: Máxima desviación puntual.
        lag_estimate: Retardo estimado en muestras.
        signal_distortion: Distorsión de nivel.
        is_effective: True si el filtro reduce ruido sin distorsionar.
        extra: Campos adicionales (extensibilidad).
    """

    filter_name: str = "none"
    n_points: int = 0
    noise_reduction_ratio: float = 0.0
    mean_absolute_error: float = 0.0
    max_absolute_error: float = 0.0
    lag_estimate: int = 0
    signal_distortion: float = 0.0
    is_effective: bool = False
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "filter_name": self.filter_name,
            "n_points": self.n_points,
            "noise_reduction_ratio": round(self.noise_reduction_ratio, 6),
            "mean_absolute_error": round(self.mean_absolute_error, 8),
            "max_absolute_error": round(self.max_absolute_error, 8),
            "lag_estimate": self.lag_estimate,
            "signal_distortion": round(self.signal_distortion, 6),
            "is_effective": self.is_effective,
        }
        if self.extra:
            d["extra"] = dict(self.extra)
        return d

    @classmethod
    def empty(cls) -> FilterSnapshot:
        return cls()

    @classmethod
    def from_dict(cls, data: dict) -> FilterSnapshot:
        """Reconstruye desde un dict serializado."""
        return cls(
            filter_name=data.get("filter_name", "none"),
            n_points=data.get("n_points", 0),
            noise_reduction_ratio=data.get("noise_reduction_ratio", 0.0),
            mean_absolute_error=data.get("mean_absolute_error", 0.0),
            max_absolute_error=data.get("max_absolute_error", 0.0),
            lag_estimate=data.get("lag_estimate", 0),
            signal_distortion=data.get("signal_distortion", 0.0),
            is_effective=data.get("is_effective", False),
            extra=data.get("extra", {}),
        )
