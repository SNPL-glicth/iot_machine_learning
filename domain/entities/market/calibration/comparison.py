"""FASE 10.5 — Calibration Comparison and Result Dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .verdicts import CalibrationVerdict, FallbackLevel
from .context_calibrator import ContextKey


@dataclass(frozen=True, slots=True)
class CalibrationComparison:
    """Comparación raw vs calibrated para un contexto."""
    
    context: str
    n_train: int
    n_val: int
    n_test: int
    
    # Métricas RAW
    raw_brier: float
    raw_ece: float
    raw_log_loss: float
    raw_wilson_lb: float
    raw_economic_edge: float
    
    # Métricas CALIBRATED
    calibrated_brier: float
    calibrated_ece: float
    calibrated_log_loss: float
    calibrated_wilson_lb: float
    calibrated_economic_edge: float
    
    # Diferencias
    brier_improvement: float
    ece_improvement: float
    log_loss_improvement: float
    wilson_improvement: float
    economic_impact: float
    
    # Veredicto
    verdict: CalibrationVerdict
    rejection_reason: Optional[str]
    
    @property
    def is_improvement(self) -> bool:
        """True si mejora Brier sin empeorar significativamente lo económico."""
        return (self.brier_improvement > 0 and 
                self.economic_impact >= -0.001)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Resultado de aplicar calibración con metadata completa."""
    
    prob_raw: float
    prob_calibrated: float
    context: ContextKey
    fallback_level: FallbackLevel
    calibrator_version: Optional[str]
    is_available: bool  # False = CALIBRATION = UNAVAILABLE
    
    @property
    def is_calibrated(self) -> bool:
        """True si se aplicó calibración."""
        return self.is_available and self.fallback_level != FallbackLevel.UNAVAILABLE