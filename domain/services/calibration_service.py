"""FASE 10.5 — Calibration Service: Evidence Gate + Calibration Application.

Este servicio implementa el "Evidence Gate" que decide:
- Si hay calibrador disponible para el contexto
- Si se debe aplicar calibración o NO TRADE
- Versionado completo por predicción

Pipeline:
    MARKET → RAW MODEL → CALIBRATOR → EVIDENCE GATE
                                          ↙         ↘
                                    NO TRADE    PREDICTION
                                                      ↓
                                                OUTCOME
                                                      ↓
                                             EVALUATION
                                                      ↓
                                                REWARD
                                                      ↓
                                ┌───────────────────┴───────────────────┐
                                ↓                                       ↓
                           PERFORMANCE                             CALIBRATION
                                ↓                                       ↓
                                └───────────────────┬───────────────────┘
                                                    ↓
                                            EXPERIMENT PROPOSAL
                                                    ↓
                                             OOS EVALUATION
                                                    ↓
                                            ACCEPT / REJECT
                                                    ↓
                                           NEW VERSION
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from iot_machine_learning.domain.entities.market.calibration import (
    AdaptiveCalibrator,
    CalibrationMethod,
    CalibrationResult,
    CalibrationVerdict,
    ContextKey,
    FallbackLevel,
    compute_economic_edge,
    compute_wilson_lb,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.calibrator_repository_v2 import (
    CalibratorRepositoryV2,
    PredictionVersioning,
)


class EvidenceGateDecision(Enum):
    """Decisión del Evidence Gate."""
    
    CALIBRATED = "calibrated"  # Calibración aplicada
    RAW = "raw"  # Sin calibrador, usar raw
    NO_TRADE = "no_trade"  # Sin calibración disponible, no operar


@dataclass(frozen=True, slots=True)
class CalibrationContext:
    """Contexto completo para calibración."""
    
    symbol: str
    strategy: str
    horizon_seconds: int
    regime: str
    model_version: str
    strategy_version: str
    evidence_policy_version: str


@dataclass(frozen=True, slots=True)
class CalibratedPrediction:
    """Predicción con calibración aplicada y versionado completo."""
    
    prediction_id: str
    symbol: str
    prob_raw: float
    prob_calibrated: float
    calibration_applied: bool
    fallback_level: FallbackLevel
    calibrator_version: str | None
    evidence_gate_decision: EvidenceGateDecision
    
    # Versionado completo
    model_version: str
    calibrator_version_full: str | None  # calibrator_v1, etc.
    strategy_version: str
    evidence_policy_version: str
    
    # Metadata
    context: CalibrationContext
    
    @property
    def is_calibrated(self) -> bool:
        return self.calibration_applied
    
    @property
    def is_available(self) -> bool:
        return self.evidence_gate_decision != EvidenceGateDecision.NO_TRADE


class CalibrationService:
    """Servicio de calibración con Evidence Gate y versionado completo."""
    
    def __init__(
        self,
        db_connection,
        min_confidence_threshold: float = 0.55,
        require_calibration: bool = True,  # Si True, NO_TRADE si no hay calibrador
    ) -> None:
        self._db = db_connection
        self._repo = CalibratorRepositoryV2(db_connection)
        self._min_confidence = min_confidence_threshold
        self._require_calibration = require_calibration
        self._adaptive_calibrator: Optional[AdaptiveCalibrator] = None
        self._calibrators_cache: dict[FallbackLevel, any] = {}
        self._current_calibrator_version: str | None = None
    
    def load_active_calibrator(self) -> bool:
        """Carga el calibrador activo desde BD."""
        active = self._repo.get_active_calibrator_v2()
        
        if not active or active.verdict != CalibrationVerdict.ACCEPTED:
            return False
        
        # Reconstruir calibradores por fallback level
        # En una implementación completa, cada fallback level tendría su propio calibrador
        # Por ahora usamos el calibrador principal
        self._current_calibrator_version = active.calibrator_id
        
        # Crear AdaptiveCalibrator con los parámetros del calibrador activo
        method = active.method
        self._adaptive_calibrator = AdaptiveCalibrator(method=method)
        
        # Restaurar calibradores desde params_json
        self._restore_calibrators(active.params_json, method)
        
        return True
    
    def _restore_calibrators(self, params_json: dict, method: CalibrationMethod) -> None:
        """Restaura calibradores por fallback level desde params_json."""
        from iot_machine_learning.domain.entities.market.calibration import ContextCalibrator, CalibrationParams
        
        # Los calibradores se guardan por contexto, necesitamos separarlos por fallback level
        # Esta es una simplificación; en producción cada nivel tendría su propia tabla
        context_calibrator = ContextCalibrator(method=method)
        
        for context_str, params_data in params_json.items():
            parts = context_str.split("·")
            if len(parts) >= 3:
                strategy = parts[0]
                horizon = int(parts[1].replace("s", ""))
                regime = parts[2]
                context = ContextKey(strategy=strategy, horizon_seconds=horizon, regime=regime)
                
                params = CalibrationParams(
                    method=CalibrationMethod(params_data["method"]),
                    params=tuple(params_data["params"]),
                    n_train=params_data["n_train"],
                    train_brier=params_data["train_brier"],
                    train_ece=params_data["train_ece"],
                )
                context_calibrator._params[context] = params
        
        # Asignar a todos los niveles de fallback (simplificado)
        self._calibrators_cache = {
            FallbackLevel.CONTEXT: context_calibrator,
            FallbackLevel.REGIME: context_calibrator,
            FallbackLevel.STRATEGY: context_calibrator,
            FallbackLevel.GLOBAL: context_calibrator,
        }
        
        self._adaptive_calibrator._calibrators = self._calibrators_cache
    
    def apply_calibration(
        self,
        context: CalibrationContext,
        prob_raw: float,
    ) -> CalibratedPrediction:
        """Aplica calibración con Evidence Gate.
        
        Returns:
            CalibratedPrediction con decisión del Evidence Gate y versionado completo.
        """
        prediction_id = str(uuid.uuid4())[:8]
        
        # Crear ContextKey
        ctx_key = ContextKey(
            strategy=context.strategy,
            horizon_seconds=context.horizon_seconds,
            regime=context.regime,
        )
        
        # Verificar si tenemos calibrador activo
        if not self._adaptive_calibrator or not self._calibrators_cache:
            # Sin calibrador activo
            decision = EvidenceGateDecision.RAW if not self._require_calibration else EvidenceGateDecision.NO_TRADE
            
            return CalibratedPrediction(
                prediction_id=prediction_id,
                symbol=context.symbol,
                prob_raw=prob_raw,
                prob_calibrated=prob_raw,
                calibration_applied=False,
                fallback_level=FallbackLevel.UNAVAILABLE,
                calibrator_version=None,
                evidence_gate_decision=decision,
                model_version=context.model_version,
                calibrator_version_full=None,
                strategy_version=context.strategy_version,
                evidence_policy_version=context.evidence_policy_version,
                context=context,
            )
        
        # Aplicar calibración con fallback
        result = self._adaptive_calibrator.apply_with_fallback(
            context=ctx_key,
            prob_raw=prob_raw,
            available_calibrators=self._calibrators_cache,
            calibrator_version=self._current_calibrator_version,
        )
        
        # Evidence Gate Decision
        if not result.is_available:
            # CALIBRATION = UNAVAILABLE
            decision = EvidenceGateDecision.RAW if not self._require_calibration else EvidenceGateDecision.NO_TRADE
        else:
            decision = EvidenceGateDecision.CALIBRATED
        
        # Verificar threshold de confianza
        final_prob = result.prob_calibrated if result.is_available else prob_raw
        if final_prob < self._min_confidence_threshold:
            decision = EvidenceGateDecision.NO_TRADE
        
        calibrated_pred = CalibratedPrediction(
            prediction_id=prediction_id,
            symbol=context.symbol,
            prob_raw=prob_raw,
            prob_calibrated=result.prob_calibrated,
            calibration_applied=result.is_available,
            fallback_level=result.fallback_level,
            calibrator_version=result.calibrator_version,
            evidence_gate_decision=decision,
            model_version=context.model_version,
            calibrator_version_full=self._current_calibrator_version,
            strategy_version=context.strategy_version,
            evidence_policy_version=context.evidence_policy_version,
            context=context,
        )
        
        # Persistir versionado en BD
        self._persist_prediction_versioning(calibrated_pred)
        
        return calibrated_pred
    
    def _persist_prediction_versioning(self, pred: CalibratedPrediction) -> None:
        """Persiste el versionado completo de la predicción."""
        # Necesitamos el prediction_id real de la BD
        # Esto se llamaría después de insertar la predicción en market_predictions
        # Por ahora es un placeholder
        pass
    
    def register_prediction_outcome(
        self,
        prediction_id: str,
        outcome: bool,  # True = direction correct
        actual_return: float | None = None,
    ) -> None:
        """Registra outcome para futura evaluación de calibración.
        
        Esto alimenta el loop de retroalimentación para FASE 11.
        """
        # En BD: actualizar market_predictions con direction_correct, actual_return
        # Y acumular en tabla de evaluación de calibración
        pass
    
    def evaluate_calibration_performance(
        self,
        calibrator_version: str,
        since_date: str,
    ) -> dict:
        """Evalúa performance del calibrador en producción.
        
        Returns métricas raw vs calibrated en datos OOS reales.
        """
        # Query predicciones con este calibrator_version desde since_date
        # Calcular Brier, ECE, LogLoss, Wilson, Economic edge
        # Comparar raw vs calibrated
        return {}
    
    def propose_new_calibrator(
        self,
        new_data: list[tuple[ContextKey, float, bool]],
        description: str,
    ) -> tuple[str | None, dict[str, any] | None]:
        """Propone nuevo calibrador con datos recientes (OOS evaluation).
        
        Returns:
            (calibrator_id, comparison) si ACCEPTED, (None, comparison) si REJECTED
        """
        if not self._adaptive_calibrator:
            return None, None
        
        # Entrenar y evaluar con nuevos datos
        calibrators, comparisons = self._adaptive_calibrator.train_and_evaluate(new_data)
        
        if not calibrators:
            return None, {"error": "Insufficient data for new calibrator"}
        
        # Verificar si al menos un fallback level fue ACCEPTED
        accepted = [c for c in comparisons.values() if c.verdict == CalibrationVerdict.ACCEPTED]
        
        if not accepted:
            # Todos rechazados
            return None, {ctx: {"verdict": c.verdict.value, "reason": c.rejection_reason} 
                         for ctx, c in comparisons.items()}
        
        # Guardar nuevo calibrador
        metadata = {
            "proposed_from": "production_data",
            "data_period": since_date,
            "fallback_levels": [k.value for k in calibrators.keys()],
        }
        
        calibrator_id = self._repo.save_calibrator_v2(
            calibrator=list(calibrators.values())[0],  # Principal
            comparison=accepted[0],
            description=description,
            metadata=metadata,
        )
        
        # Recargar calibrador activo
        self.load_active_calibrator()
        
        return calibrator_id, {ctx: {"verdict": c.verdict.value, "brier_improvement": c.brier_improvement} 
                              for ctx, c in comparisons.items()}


def render_evidence_gate_decision(pred: CalibratedPrediction) -> str:
    """Renderiza decisión del Evidence Gate para logging."""
    lines = [
        "EVIDENCE GATE DECISION",
        "=" * 22,
        f"Prediction ID: {pred.prediction_id}",
        f"Symbol: {pred.symbol}",
        f"Decision: {pred.evidence_gate_decision.value.upper()}",
        f"Raw Prob: {pred.prob_raw:.4f}",
        f"Calibrated Prob: {pred.prob_calibrated:.4f}",
        f"Calibration Applied: {pred.calibration_applied}",
        f"Fallback Level: {pred.fallback_level.value}",
        f"Calibrator Version: {pred.calibrator_version_full or 'N/A'}",
        "",
        "VERSIONING",
        "-" * 10,
        f"Model: {pred.model_version}",
        f"Calibrator: {pred.calibrator_version_full or 'N/A'}",
        f"Strategy: {pred.strategy_version}",
        f"Evidence Policy: {pred.evidence_policy_version}",
    ]
    
    if pred.evidence_gate_decision == EvidenceGateDecision.NO_TRADE:
        lines.append("")
        lines.append("🚫 NO TRADE — Calibración no disponible o confianza insuficiente")
    elif pred.evidence_gate_decision == EvidenceGateDecision.RAW:
        lines.append("")
        lines.append("⚠️  RAW — Usando probabilidad sin calibrar (sin calibrador válido)")
    else:
        lines.append("")
        lines.append(f"✓ CALIBRATED — Fallback: {pred.fallback_level.value}")
    
    return "\n".join(lines)