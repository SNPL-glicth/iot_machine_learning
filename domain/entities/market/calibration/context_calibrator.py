"""FASE 10.1 — Context Calibrator: calibrador por estrategia/horizonte/régimen.

Objetivo:
- Calibrar probabilidades por contexto (estrategia × horizonte × régimen)
- Corregir la sobref confianza en alta confianza y subestimación en baja
- Versionado de calibradores (calibrator_v1, calibrator_v2, etc.)
- Comparación raw vs calibrated (Brier/ECE)
- Nunca modificar predicciones históricas

Método:
- Para cada contexto, aprender función de mapeo: prob_raw → prob_calibrated
- Usar Platt scaling simplificado: prob_calibrated = sigmoid(a * prob_raw + b)
- Calibrar solo en datos de entrenamiento, evaluar out-of-sample
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Final

from .context_types import (
    CalibrationMethod,
    ContextKey,
    CalibrationParams,
    CalibrationResult,
)
from .fitting import fit_platt_scaling, fit_bucket_calibration
from .metrics import compute_brier, compute_ece


__all__ = [
    "ContextCalibrator",
]


def _platt_scale(prob: float, a: float, b: float) -> float:
    """Aplica Platt scaling: sigmoid(a * prob + b)."""
    x = a * prob + b
    # Evitar overflow en exp
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _bucket_calibration(prob: float, buckets: dict[float, float]) -> float:
    """Aplica calibración por bucket."""
    bucket = round(prob * 10) / 10
    return buckets.get(bucket, prob)  # Fallback a raw si no hay bucket


class ContextCalibrator:
    """Calibrador contextual que aprende mapeos por contexto."""
    
    def __init__(self, method: CalibrationMethod = CalibrationMethod.PLATT) -> None:
        self.method = method
        self._params: dict[ContextKey, CalibrationParams] = {}
    
    def fit(
        self,
        data: list[tuple[ContextKey, float, bool]],  # (context, prob_raw, outcome)
    ) -> None:
        """Ajusta calibración para cada contexto."""
        # Agrupar por contexto
        context_data: dict[ContextKey, list[tuple[float, bool]]] = defaultdict(list)
        for context, prob, outcome in data:
            context_data[context].append((prob, outcome))
        
        # Ajustar calibración por contexto
        for context, items in context_data.items():
            raw_probs = [prob for prob, _ in items]
            outcomes = [outcome for _, outcome in items]
            
            train_brier = compute_brier(raw_probs, outcomes)
            train_ece = compute_ece(raw_probs, outcomes)
            
            if self.method == CalibrationMethod.PLATT:
                a, b = fit_platt_scaling(raw_probs, outcomes)
                params = CalibrationParams(
                    method=self.method,
                    params=(a, b),
                    n_train=len(items),
                    train_brier=train_brier,
                    train_ece=train_ece,
                )
            elif self.method == CalibrationMethod.BUCKET:
                buckets = fit_bucket_calibration(raw_probs, outcomes)
                params = CalibrationParams(
                    method=self.method,
                    params=tuple(buckets.items()),
                    n_train=len(items),
                    train_brier=train_brier,
                    train_ece=train_ece,
                )
            else:
                continue
            
            self._params[context] = params
    
    def calibrate(
        self,
        context: ContextKey,
        prob_raw: float,
    ) -> CalibrationResult:
        """Aplica calibración a una probabilidad."""
        params = self._params.get(context)
        
        if params is None or not params.is_valid:
            return CalibrationResult(
                prob_raw=prob_raw,
                prob_calibrated=prob_raw,
                context=context,
                params=None,
            )
        
        if params.method == CalibrationMethod.PLATT:
            a, b = params.params
            prob_calibrated = _platt_scale(prob_raw, a, b)
        elif params.method == CalibrationMethod.BUCKET:
            buckets = dict(params.params)
            prob_calibrated = _bucket_calibration(prob_raw, buckets)
        else:
            prob_calibrated = prob_raw
        
        # Clipping a [0.05, 0.95] para evitar extremos
        prob_calibrated = max(0.05, min(0.95, prob_calibrated))
        
        return CalibrationResult(
            prob_raw=prob_raw,
            prob_calibrated=prob_calibrated,
            context=context,
            params=params,
        )
    
    def evaluate(
        self,
        data: list[tuple[ContextKey, float, bool]],  # (context, prob_raw, outcome)
    ) -> dict[ContextKey, dict[str, float]]:
        """Evalúa calibración out-of-sample por contexto."""
        results: dict[ContextKey, dict[str, float]] = {}
        
        # Agrupar por contexto
        context_data: dict[ContextKey, list[tuple[float, bool]]] = defaultdict(list)
        for context, prob, outcome in data:
            context_data[context].append((prob, outcome))
        
        for context, items in context_data.items():
            raw_probs = [prob for prob, _ in items]
            outcomes = [outcome for _, outcome in items]
            
            # Calibrar probabilidades
            calibrated_probs = []
            for prob in raw_probs:
                result = self.calibrate(context, prob)
                calibrated_probs.append(result.prob_calibrated)
            
            # Métricas
            raw_brier = compute_brier(raw_probs, outcomes)
            raw_ece = compute_ece(raw_probs, outcomes)
            calibrated_brier = compute_brier(calibrated_probs, outcomes)
            calibrated_ece = compute_ece(calibrated_probs, outcomes)
            
            results[context] = {
                "n": len(items),
                "raw_brier": raw_brier,
                "raw_ece": raw_ece,
                "calibrated_brier": calibrated_brier,
                "calibrated_ece": calibrated_ece,
                "brier_improvement": raw_brier - calibrated_brier,
                "ece_improvement": raw_ece - calibrated_ece,
            }
        
        return results