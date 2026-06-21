"""Registry centralizado de constantes numéricas.

Principio: Single Responsibility - solo define constantes.
Aplica SOLID: frozen dataclasses para inmutabilidad.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Final


@dataclass(frozen=True)
class EpsilonConfig:
    """Jerarquía de epsilons por uso específico.
    
    Diferentes epsilons para diferentes propósitos numéricos:
    - COMPARISON: Para |x| < eps (comparaciones de cercanía a cero)
    - DIVISION: Para denominadores (más conservador para prevenir overflow)
    - CONFIDENCE: Para cálculos de confianza (rango intermedio)
    - CORRELATION: Para correlaciones (similar a COMPARISON)
    - GRADIENT: Para gradientes (similar a COMPARISON)
    - KALMAN_R: Mínimo R para Kalman gain (previene K=1 que ignora el modelo)
    - KALMAN_P: Mínimo P para evitar filtro congelado (previene K→0 por underflow)
    """
    COMPARISON: Final[float] = 1e-9   # Para |x| < eps
    DIVISION: Final[float] = 1e-12    # Para denominadores
    CONFIDENCE: Final[float] = 1e-6   # Para cálculos de confianza
    CORRELATION: Final[float] = 1e-9  # Para correlaciones
    GRADIENT: Final[float] = 1e-9     # Para gradientes
    KALMAN_R: Final[float] = 1e-6     # Mínimo R para Kalman gain
    KALMAN_P: Final[float] = 1e-10    # Mínimo P para evitar filtro congelado


@dataclass(frozen=True)
class StatisticalThresholds:
    """Thresholds estadísticos calibrados.
    
    Z-score: 2σ ≈ 95.4%, 3σ ≈ 99.7% bajo normalidad.
    Contamination calibrado para consistencia con z-score.
    """
    # Z-score thresholds (σ bajo normalidad)
    Z_SCORE_LOWER: Final[float] = 2.0
    Z_SCORE_UPPER: Final[float] = 3.0
    
    # Contamination calibrado para consistencia con z-score
    # 3σ ≈ 0.3% bajo normalidad, usamos 0.5% como default conservador
    CONTAMINATION_DEFAULT: Final[float] = 0.005  # 0.5% (conservador)
    CONTAMINATION_MIN: Final[float] = 0.001
    CONTAMINATION_MAX: Final[float] = 0.05
    
    # IQR (Tukey fences)
    IQR_FENCE_MULTIPLIER: Final[float] = 1.5  # Tukey standard
    
    # LOF
    LOF_MAX_NEIGHBORS: Final[int] = 20


@dataclass(frozen=True)
class ConfidenceConfig:
    """Configuración unificada de confidence.

    Floor y ceiling unificados entre todos los engines.
    Temperature scaling por régimen (justificación en temperature_scaling.py).
    """
    MIN_CONFIDENCE: Final[float] = 0.5  # Floor unificado (0.3→0.5 para industria)
    MAX_CONFIDENCE: Final[float] = 0.95  # Ceiling unificado
    # Temperature scaling por régimen (justificación en temperature_scaling.py)
    TEMP_STABLE: Final[float] = 1.2  # ligera suavización, datos predecibles
    TEMP_TRENDING: Final[float] = 1.5  # moderada, tendencia añade incertidumbre
    TEMP_VOLATILE: Final[float] = 2.0  # alta, volatilidad = alta incertidumbre
    TEMP_NOISY: Final[float] = 1.8  # alta, ruido degrada confianza
    TEMP_DEFAULT: Final[float] = 1.5  # igual que TRENDING
    
    def validate(self, value: float) -> float:
        """Valida y clampea confidence al rango.
        
        Args:
            value: Confidence a validar.
        
        Returns:
            Confidence clampeada a [MIN_CONFIDENCE, MAX_CONFIDENCE].
        """
        return max(self.MIN_CONFIDENCE, min(self.MAX_CONFIDENCE, value))


@dataclass(frozen=True)
class _InhibitionThresholds:
    """
    Thresholds para InhibitionGate.
    
    Justificación estadística:
    - STABILITY: 0.6 = requiere 60% de estabilidad mínima.
      Derivado de: si error_rate > 40% el engine es poco confiable.
    - FIT_ERROR: 5.0 = MSE normalizado máximo aceptable.
      Derivado de: >5.0 error en fit indica modelo no converge.
    - RECENT_ERROR: 10.0 = error reciente máximo antes de inhibición.
      Derivado de: 2x FIT_ERROR para tolerar varianza temporal.
    
    Estos valores son heurísticos calibrados empíricamente.
    Para calibración por dominio usar DomainThresholdConfig.
    """
    STABILITY: float = 0.6
    FIT_ERROR: float = 5.0
    RECENT_ERROR: float = 10.0


@dataclass(frozen=True)
class _PenaltyThresholds:
    """
    Thresholds para penalty-based confidence calibration.
    
    Usados por domain/services/confidence_calibrator.py para determinar
    cuándo aplicar penalidades por calidad de datos.
    
    Justificación empírica:
    - MIN_POINTS: 10 = mínimo para estadística robusta (n≥10 regla general)
    - MAX_NOISE_RATIO: 0.6 = σ/|μ| > 60% indica señal muy ruidosa
    - MAX_ENGINE_DISAGREEMENT: 0.3 = diferencia >30% entre engines indica incertidumbre
    """
    MIN_POINTS: Final[int] = 10  # n_points < MIN_POINTS → penalty
    MAX_NOISE_RATIO: Final[float] = 0.6  # noise_ratio > MAX_NOISE_RATIO → penalty
    MAX_ENGINE_DISAGREEMENT: Final[float] = 0.3  # disagreement > MAX_ENGINE_DISAGREEMENT → penalty


# Singleton instances (single source of truth)
EPSILON = EpsilonConfig()
STAT_THRESHOLDS = StatisticalThresholds()
CONFIDENCE = ConfidenceConfig()
INHIBITION_THRESHOLDS = _InhibitionThresholds()
PENALTY_THRESHOLDS = _PenaltyThresholds()
