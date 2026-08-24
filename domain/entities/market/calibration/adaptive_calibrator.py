"""FASE 10.5 — Adaptive Calibration Profesional: train/val/test sin leakage.

Condiciones estrictas:
1. NO leakage: el calibrador NO puede aprender del mismo dato que evalúa
2. Comparación obligatoria raw vs calibrated (Brier/ECE/LogLoss/Wilson/Economic)
3. Versionado real: model/calibrator/strategy/evidence versions por predicción
4. Sistema de rechazo: calibradores pueden ser rechazados si empeoran
5. Fallback hierarchy: context→regime→strategy→global→unavailable

Pipeline estricto:
    TRAIN → calibrator_v1 → VALIDATION → ¿mejoró? → TEST congelado → ACCEPT/REJECT
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .context_calibrator import (
    CalibrationMethod,
    ContextCalibrator,
    ContextKey,
)
from .verdicts import CalibrationVerdict, FallbackLevel
from .comparison import CalibrationComparison, CalibrationResult
from .split import train_val_test_split
from .fallback import (
    build_fallback_calibrators,
    get_fallback_calibrator,
    resolve_fallback_context,
)
from .evaluation import evaluate_all_fallback_levels
from .render import render_calibration_comparison


__all__ = [
    "CalibrationVerdict",
    "CalibrationComparison",
    "CalibrationResult",
    "AdaptiveCalibrator",
    "FallbackLevel",
    "train_val_test_split",
    "render_calibration_comparison",
]


class AdaptiveCalibrator:
    """Calibrador adaptativo con train/val/test split y rechazo."""
    
    def __init__(
        self,
        method: CalibrationMethod = CalibrationMethod.PLATT,
        min_train_samples: int = 100,
        min_val_samples: int = 50,
        min_test_samples: int = 50,
        brier_tolerance: float = 0.0,
        economic_tolerance: float = -0.001,
        min_context_samples: int = 30,
    ) -> None:
        self.method = method
        self.min_train_samples = min_train_samples
        self.min_val_samples = min_val_samples
        self.min_test_samples = min_test_samples
        self.brier_tolerance = brier_tolerance
        self.economic_tolerance = economic_tolerance
        self.min_context_samples = min_context_samples
        
        self._calibrators: Dict[FallbackLevel, ContextCalibrator] = {}
        self._comparisons: Dict[str, CalibrationComparison] = {}
        self._version: Optional[str] = None
    
    def train_and_evaluate(
        self,
        data: List[Tuple[ContextKey, float, bool]],
        context_key: Optional[ContextKey] = None,
    ) -> Tuple[Optional[Dict[FallbackLevel, ContextCalibrator]], Optional[Dict[str, CalibrationComparison]]]:
        """Entrena y evalúa calibradores con train/val/test split para todos los fallback levels."""
        
        if len(data) < (self.min_train_samples + self.min_val_samples + self.min_test_samples):
            return None, None
        
        # Train/val/test split temporal
        train_data, val_data, test_data = train_val_test_split(data)
        
        if len(train_data) < self.min_train_samples:
            return None, None
        if len(val_data) < self.min_val_samples:
            return None, None
        if len(test_data) < self.min_test_samples:
            return None, None
        
        # Construir calibradores por nivel de fallback (SOLO con train)
        calibrators = build_fallback_calibrators(
            train_data, self.method, self.min_context_samples
        )
        
        if not calibrators:
            return None, None
        
        # Evaluar cada calibrador en validation y test
        comparisons = evaluate_all_fallback_levels(
            calibrators=calibrators,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            min_val_samples=self.min_val_samples,
            min_test_samples=self.min_test_samples,
            brier_tolerance=self.brier_tolerance,
            economic_tolerance=self.economic_tolerance,
        )

        # Swap conservador (FASE 10.5, no-negociable 4): el candidato solo
        # se activa si TODO lo evaluado fue aceptado. Un solo nivel
        # rechazado, o cero comparaciones (nada alcanzó val/test mínimos),
        # deja la versión anterior de _calibrators intacta y devuelve None
        # como señal de "sin cambio de estado". Las comparaciones del
        # candidato se registran igual: son diagnóstico, no estado activo.
        if not comparisons or any(
            comp.verdict != CalibrationVerdict.ACCEPTED
            for comp in comparisons.values()
        ):
            self._comparisons = comparisons or self._comparisons
            return None, comparisons

        # Guardar calibradores y comparaciones
        self._calibrators = calibrators
        self._comparisons = comparisons

        return calibrators, comparisons
    
    def get_fallback_calibrator(
        self,
        context: ContextKey,
        available_calibrators: Optional[Dict[FallbackLevel, ContextCalibrator]] = None,
    ) -> Tuple[FallbackLevel, Optional[ContextCalibrator]]:
        """Obtiene calibrador según fallback hierarchy."""
        return get_fallback_calibrator(context, available_calibrators or self._calibrators)
    
    def apply_with_fallback(
        self,
        context: ContextKey,
        prob_raw: float,
        available_calibrators: Optional[Dict[FallbackLevel, ContextCalibrator]] = None,
        calibrator_version: Optional[str] = None,
    ) -> CalibrationResult:
        """Aplica calibración con fallback hierarchy.
        
        Returns CalibrationResult with is_available=False if no calibrator available.
        """
        fallback_level, calibrator = self.get_fallback_calibrator(context, available_calibrators)
        
        if calibrator is None or fallback_level == FallbackLevel.UNAVAILABLE:
            return CalibrationResult(
                prob_raw=prob_raw,
                prob_calibrated=prob_raw,
                context=context,
                fallback_level=FallbackLevel.UNAVAILABLE,
                calibrator_version=calibrator_version,
                is_available=False,
            )

        # La clave de aplicación es la gruesa del nivel seleccionado: el
        # calibrador ajustó sus params bajo esa clave (ver
        # resolve_fallback_context). Con el contexto fino original los
        # niveles no-CONTEXT hacían miss y devolvían la prob cruda sin
        # marcar UNCALIBRATED.
        fit_context = resolve_fallback_context(context, fallback_level)
        result = calibrator.calibrate(fit_context, prob_raw)
        return CalibrationResult(
            prob_raw=prob_raw,
            prob_calibrated=result.prob_calibrated,
            context=fit_context,
            fallback_level=fallback_level,
            calibrator_version=calibrator_version,
            is_available=True,
        )
    
    def get_comparisons(self) -> Dict[str, CalibrationComparison]:
        """Obtiene todas las comparaciones de evaluación."""
        return self._comparisons
    
    def set_version(self, version: str) -> None:
        """Establece versión del calibrador."""
        self._version = version
    
    def get_version(self) -> Optional[str]:
        """Obtiene versión del calibrador."""
        return self._version