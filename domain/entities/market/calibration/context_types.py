"""FASE 10.1 — Context Calibration Types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CalibrationMethod(Enum):
    """Métodos de calibración."""
    
    NONE = "none"  # Sin calibración (raw)
    PLATT = "platt"  # Platt scaling: sigmoid(a * x + b)
    ISOTONIC = "isotonic"  # Isotonic regression (requiere más datos)
    BUCKET = "bucket"  # Bucket calibration (como el observatory)


@dataclass(frozen=True, slots=True)
class ContextKey:
    """Clave única de contexto para calibración."""
    
    strategy: str
    horizon_seconds: int
    regime: str
    
    def __str__(self) -> str:
        regime = self.regime or "ALL"
        return f"{self.strategy}·{self.horizon_seconds}s·{regime}"
    
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ContextKey):
            return NotImplemented
        return (self.strategy, self.horizon_seconds, self.regime) < (other.strategy, other.horizon_seconds, other.regime)


@dataclass(frozen=True, slots=True)
class CalibrationParams:
    """Parámetros de calibración para un contexto."""
    
    method: CalibrationMethod
    params: tuple[float, ...]  # (a, b) para Platt, buckets para bucket method
    n_train: int  # Muestras usadas para entrenar
    train_brier: float  # Brier score en entrenamiento
    train_ece: float  # ECE en entrenamiento
    
    @property
    def is_valid(self) -> bool:
        """True si los parámetros son válidos (mínimo de datos)."""
        return self.n_train >= 20  # Mínimo 20 muestras para calibrar


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Resultado de aplicar calibración a una probabilidad."""
    
    prob_raw: float
    prob_calibrated: float
    context: ContextKey
    params: CalibrationParams | None
    
    @property
    def is_calibrated(self) -> bool:
        """True si se aplicó calibración."""
        return self.params is not None