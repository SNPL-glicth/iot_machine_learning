"""Puerto abstracto para calibración de confianza probabilística.

Responsabilidad: Definir interfaz para calibrar confidence scores
a probabilidades reales bien calibradas (ej: 0.85 = 85% de acierto).

Zero dependencias de sklearn, numpy, o cualquier librería externa aquí.
La implementación concreta puede usar Platt Scaling, Isotonic Regression,
o cualquier método en la capa de Infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CalibrationStats:
    """Estadísticas de calibración.
    
    Attributes:
        n_samples: Número de muestras usadas para calibrar.
        ece: Expected Calibration Error (0.0 = perfecto, >0.15 = mal calibrado).
        reliability: Diccionario por bin: {bin_id: (confianza_promedio, accuracy_real)}.
        is_ready: True si hay suficientes muestras para calibrar confiablemente.
        calibrator_type: Tipo de calibrador usado (platt, isotonic, null).
    """
    
    n_samples: int
    ece: float
    reliability: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    is_ready: bool = False
    calibrator_type: str = "unknown"


@dataclass(frozen=True)
class CalibratedScore:
    """Resultado de calibración de un score individual.
    
    Attributes:
        calibrated_score: Score calibrado [0.0, 1.0].
        raw_score: Score original sin calibrar.
        calibration_applied: True si se aplicó calibración (n_samples >= min_samples).
        calibration_delta: Diferencia (calibrated - raw).
    """
    
    calibrated_score: float
    raw_score: float
    calibration_applied: bool = False
    calibration_delta: float = 0.0


class ConfidenceCalibratorPort(ABC):
    """Puerto abstracto para calibración de confianza probabilística.
    
    Implementaciones concretas:
    - PlattCalibrator: Platt Scaling con SGD online (infrastructure)
    - IsotonicCalibrator: Isotonic Regression con warm updates (infrastructure)
    - RegimeAwareCalibrator: Selecciona método según régimen (infrastructure)
    - NullCalibrator: Pass-through, devuelve score sin modificar (domain)
    
    Design notes:
    - Todos los métodos son fail-safe: nunca lanzan excepciones al caller.
    - El calibrador aprende online: update() se llama con cada nuevo dato.
    - Thread-safety es responsabilidad de la implementación.
    """
    
    @abstractmethod
    def calibrate(self, raw_score: float) -> CalibratedScore:
        """Calibrar un score crudo a probabilidad bien calibrada.
        
        Args:
            raw_score: Score de confianza crudo [0.0, 1.0].
            
        Returns:
            CalibratedScore con score calibrado y metadatos.
            Si el calibrador no está listo (is_ready=False), 
            devuelve raw_score sin modificar.
        """
        ...
    
    @abstractmethod
    def update(self, raw_score: float, actual_outcome: float) -> None:
        """Actualizar el calibrador con un resultado real (online learning).
        
        Args:
            raw_score: Score de confianza que se reportó para la predicción.
            actual_outcome: Resultado real observado [0.0, 1.0].
                Para clasificación binaria: 1.0 = correcto, 0.0 = incorrecto.
                Para error continuo: 1.0 - normalized_error.
        """
        ...
    
    @abstractmethod
    def is_ready(self) -> bool:
        """True cuando hay suficientes muestras para calibrar confiablemente.
        
        Returns:
            True si n_samples >= min_samples (configurable, default 50).
        """
        ...
    
    @abstractmethod
    def get_stats(self) -> CalibrationStats:
        """Obtener estadísticas de calibración actuales.
        
        Returns:
            CalibrationStats con ECE, n_samples, reliability por bins.
        """
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """Resetear el calibrador a estado inicial (olvidar todo el historial)."""
        ...


class NullCalibrator(ConfidenceCalibratorPort):
    """Calibrador no-op que devuelve scores sin modificar.
    
    Usado como default cuando la calibración está desactivada o cuando
    no hay suficientes datos todavía.
    """
    
    def calibrate(self, raw_score: float) -> CalibratedScore:
        """Pass-through: devuelve raw_score sin modificar."""
        return CalibratedScore(
            calibrated_score=max(0.0, min(1.0, raw_score)),
            raw_score=raw_score,
            calibration_applied=False,
            calibration_delta=0.0,
        )
    
    def update(self, raw_score: float, actual_outcome: float) -> None:
        """No-op: no acumula historial."""
        pass
    
    def is_ready(self) -> bool:
        """Siempre False: nunca está listo porque no hace nada."""
        return False
    
    def get_stats(self) -> CalibrationStats:
        """Devuelve stats vacías con ECE=0.0."""
        return CalibrationStats(
            n_samples=0,
            ece=0.0,
            reliability={},
            is_ready=False,
            calibrator_type="null",
        )
    
    def reset(self) -> None:
        """No-op: no hay nada que resetear."""
        pass
