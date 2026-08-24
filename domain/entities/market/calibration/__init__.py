"""FASE 10.1 + 10.5 — Calibration Module: calibración contextual profesional.

Este módulo implementa calibración de probabilidades por contexto
(estrategia × horizonte × régimen) con condiciones estrictas de FASE 10.5:

- Train/val/test split SIN leakage
- Comparación obligatoria raw vs calibrated (Brier/ECE/LogLoss/Wilson/Economic)
- Sistema de rechazo de calibradores
- Fallback hierarchy (context→regime→strategy→global→unavailable)
- Versionado real (model/calibrator/strategy/evidence versions)

Componentes:
- ContextCalibrator: calibrador básico por contexto
- AdaptiveCalibrator: calibrador con train/val/test y rechazo
- CalibrationMethod: métodos de calibración (Platt, bucket, isotonic)
- CalibrationComparison: comparación raw vs calibrated
- FallbackLevel: niveles de fallback

Reglas estrictas (FASE 10.5):
1. NO leakage: el calibrador NO puede aprender del mismo dato que evalúa
2. Comparación obligatoria raw vs calibrated
3. Versionado real por predicción
4. Sistema de rechazo: calibradores pueden ser rechazados
5. Fallback hierarchy cuando no hay evidencia suficiente
"""

from .context_types import (
    CalibrationMethod,
    CalibrationParams,
    ContextKey,
)
from .context_calibrator import (
    ContextCalibrator,
    compute_brier,
    compute_ece,
)
from .factory import (
    fit_context_calibrator,
    apply_calibration,
)

from .adaptive_calibrator import (
    AdaptiveCalibrator,
    CalibrationComparison,
    CalibrationResult,
    CalibrationVerdict,
    FallbackLevel,
    render_calibration_comparison,
    train_val_test_split,
)

from .metrics import (
    compute_log_loss,
    compute_wilson_lb,
    compute_economic_edge,
)

from .verdicts import CalibrationVerdict as CalibrationVerdictEnum, FallbackLevel as FallbackLevelEnum

from .comparison import CalibrationComparison as CalibrationComparisonClass, CalibrationResult as CalibrationResultClass

from .split import train_val_test_split as train_val_test_split_func

from .fallback import build_fallback_calibrators, get_fallback_calibrator

from .evaluation import evaluate_context, evaluate_all_fallback_levels

from .render import render_calibration_comparison as render_calibration_comparison_func

from .pipeline import (
    CalibrationEvidence,
    CalibratedPredictor,
    collect_training_pairs,
    export_calibrator_state,
    import_calibrator_state,
    try_refit,
    wrap_predictor,
)

from .gate import (
    EvidenceGate,
    GateReason,
    PaperDecision,
    TradeAction,
)

__all__ = [
    # ContextCalibrator (FASE 10.1)
    "CalibrationMethod",
    "CalibrationParams",
    "ContextCalibrator",
    "ContextKey",
    "apply_calibration",
    "compute_brier",
    "compute_ece",
    "fit_context_calibrator",
    # AdaptiveCalibrator (FASE 10.5) - Main classes
    "AdaptiveCalibrator",
    "CalibrationComparison",
    "CalibrationResult",
    "CalibrationVerdict",
    "FallbackLevel",
    # Pipeline integration (FASE 10.5 cierre)
    "CalibrationEvidence",
    "CalibratedPredictor",
    "EvidenceGate",
    "GateReason",
    "PaperDecision",
    "TradeAction",
    "collect_training_pairs",
    "export_calibrator_state",
    "import_calibrator_state",
    "try_refit",
    "wrap_predictor",
    # Metrics
    "compute_log_loss",
    "compute_wilson_lb",
    "compute_economic_edge",
    # Split
    "train_val_test_split",
    # Render
    "render_calibration_comparison",
    # Submodules (for advanced usage)
    "build_fallback_calibrators",
    "get_fallback_calibrator",
    "evaluate_context",
    "evaluate_all_fallback_levels",
]