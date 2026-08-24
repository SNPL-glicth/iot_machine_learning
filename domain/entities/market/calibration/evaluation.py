"""FASE 10.5 — Calibration Evaluation Logic."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Dict, List, Tuple

from .context_calibrator import (
    ContextCalibrator, ContextKey, compute_brier, compute_ece
)
from .metrics import compute_log_loss, compute_wilson_lb, compute_economic_edge
from .verdicts import CalibrationVerdict, FallbackLevel
from .comparison import CalibrationComparison


def evaluate_context(
    calibrator: ContextCalibrator,
    context: ContextKey,
    val_probs: List[float],
    val_outcomes: List[bool],
    test_probs: List[float],
    test_outcomes: List[bool],
    brier_tolerance: float,
    economic_tolerance: float,
) -> CalibrationComparison:
    """Evalúa un contexto específico."""
    
    # Métricas RAW (test)
    raw_brier = compute_brier(test_probs, test_outcomes)
    raw_ece = compute_ece(test_probs, test_outcomes)
    raw_log_loss = compute_log_loss(test_probs, test_outcomes)
    raw_wilson_lb = compute_wilson_lb(sum(1 for o in test_outcomes if o), len(test_outcomes))
    raw_economic_edge = compute_economic_edge(test_probs, test_outcomes)
    
    # Calibrar probabilidades de test
    calibrated_test_probs = []
    for prob in test_probs:
        result = calibrator.calibrate(context, prob)
        calibrated_test_probs.append(result.prob_calibrated)
    
    # Métricas CALIBRATED (test)
    calibrated_brier = compute_brier(calibrated_test_probs, test_outcomes)
    calibrated_ece = compute_ece(calibrated_test_probs, test_outcomes)
    calibrated_log_loss = compute_log_loss(calibrated_test_probs, test_outcomes)
    calibrated_wilson_lb = compute_wilson_lb(sum(1 for o in test_outcomes if o), len(test_outcomes))
    calibrated_economic_edge = compute_economic_edge(calibrated_test_probs, test_outcomes)
    
    # Diferencias
    brier_improvement = raw_brier - calibrated_brier
    ece_improvement = raw_ece - calibrated_ece
    log_loss_improvement = raw_log_loss - calibrated_log_loss
    wilson_improvement = calibrated_wilson_lb - raw_wilson_lb
    economic_impact = calibrated_economic_edge - raw_economic_edge
    
    # Veredicto (protocolo FASE 10.5): la aceptación de una calibración de
    # PROBABILIDAD se decide por mejora de Brier en test congelado. El
    # impacto económico NO puerta el veredicto: un modelo crudo
    # sobreconfiado puede tener pseudo-edge por suerte direccional del
    # segmento test, y exigir no empeorarlo rechazaría recalibraciones
    # honestas (se observó REJECTED con Brier +0.15). La métrica se calcula
    # y expone para dashboards; la calibración económica es otra fase.
    if brier_improvement > brier_tolerance:
        verdict = CalibrationVerdict.ACCEPTED
        rejection_reason = None
    else:
        verdict = CalibrationVerdict.REJECTED
        rejection_reason = f"Brier no mejoró: {brier_improvement:+.4f}"
    
    return CalibrationComparison(
        context=str(context),
        n_train=0,  # Will be filled by caller
        n_val=len(val_probs),
        n_test=len(test_probs),
        raw_brier=raw_brier,
        raw_ece=raw_ece,
        raw_log_loss=raw_log_loss,
        raw_wilson_lb=raw_wilson_lb,
        raw_economic_edge=raw_economic_edge,
        calibrated_brier=calibrated_brier,
        calibrated_ece=calibrated_ece,
        calibrated_log_loss=calibrated_log_loss,
        calibrated_wilson_lb=calibrated_wilson_lb,
        calibrated_economic_edge=calibrated_economic_edge,
        brier_improvement=brier_improvement,
        ece_improvement=ece_improvement,
        log_loss_improvement=log_loss_improvement,
        wilson_improvement=wilson_improvement,
        economic_impact=economic_impact,
        verdict=verdict,
        rejection_reason=rejection_reason,
    )


def evaluate_all_fallback_levels(
    calibrators: Dict[FallbackLevel, ContextCalibrator],
    train_data: List[Tuple[ContextKey, float, bool]],
    val_data: List[Tuple[ContextKey, float, bool]],
    test_data: List[Tuple[ContextKey, float, bool]],
    min_val_samples: int,
    min_test_samples: int,
    brier_tolerance: float,
    economic_tolerance: float,
) -> Dict[str, CalibrationComparison]:
    """Evalúa cada calibrador en validation y test para todos los fallback levels."""
    
    comparisons: Dict[str, CalibrationComparison] = {}
    
    # Agrupar val y test por contexto
    val_grouped: Dict[ContextKey, List[Tuple[float, bool]]] = defaultdict(list)
    test_grouped: Dict[ContextKey, List[Tuple[float, bool]]] = defaultdict(list)
    
    for context, prob, outcome in val_data:
        val_grouped[context].append((prob, outcome))
    for context, prob, outcome in test_data:
        test_grouped[context].append((prob, outcome))
    
    # Evaluar CONTEXT level
    if FallbackLevel.CONTEXT in calibrators:
        cal = calibrators[FallbackLevel.CONTEXT]
        for context in val_grouped.keys():
            if context not in test_grouped:
                continue
            if len(val_grouped[context]) < min_val_samples:
                continue
            if len(test_grouped[context]) < min_test_samples:
                continue
            
            val_probs, val_outcomes = zip(*val_grouped[context])
            test_probs, test_outcomes = zip(*test_grouped[context])
            
            comp = evaluate_context(cal, context, list(val_probs), list(val_outcomes),
                                    list(test_probs), list(test_outcomes),
                                    brier_tolerance, economic_tolerance)
            comp = CalibrationComparison(
                context=comp.context,
                n_train=len(train_data),
                n_val=comp.n_val,
                n_test=comp.n_test,
                raw_brier=comp.raw_brier,
                raw_ece=comp.raw_ece,
                raw_log_loss=comp.raw_log_loss,
                raw_wilson_lb=comp.raw_wilson_lb,
                raw_economic_edge=comp.raw_economic_edge,
                calibrated_brier=comp.calibrated_brier,
                calibrated_ece=comp.calibrated_ece,
                calibrated_log_loss=comp.calibrated_log_loss,
                calibrated_wilson_lb=comp.calibrated_wilson_lb,
                calibrated_economic_edge=comp.calibrated_economic_edge,
                brier_improvement=comp.brier_improvement,
                ece_improvement=comp.ece_improvement,
                log_loss_improvement=comp.log_loss_improvement,
                wilson_improvement=comp.wilson_improvement,
                economic_impact=comp.economic_impact,
                verdict=comp.verdict,
                rejection_reason=comp.rejection_reason,
            )
            comparisons[str(context)] = comp
    
    # Evaluar HORIZON level
    if FallbackLevel.HORIZON in calibrators:
        cal = calibrators[FallbackLevel.HORIZON]
        horizon_val: Dict[Tuple[str, int], List[Tuple[float, bool]]] = defaultdict(list)
        horizon_test: Dict[Tuple[str, int], List[Tuple[float, bool]]] = defaultdict(list)

        for context, prob, outcome in val_data:
            horizon_val[(context.strategy, context.horizon_seconds)].append((prob, outcome))
        for context, prob, outcome in test_data:
            horizon_test[(context.strategy, context.horizon_seconds)].append((prob, outcome))

        for horizon_key in horizon_val:
            if horizon_key not in horizon_test:
                continue
            if len(horizon_val[horizon_key]) < min_val_samples:
                continue
            if len(horizon_test[horizon_key]) < min_test_samples:
                continue

            val_probs, val_outcomes = zip(*horizon_val[horizon_key])
            test_probs, test_outcomes = zip(*horizon_test[horizon_key])

            context = ContextKey(horizon_key[0], horizon_key[1], "ALL")
            comp = evaluate_context(cal, context, list(val_probs), list(val_outcomes),
                                    list(test_probs), list(test_outcomes),
                                    brier_tolerance, economic_tolerance)
            comp = dataclasses.replace(comp, n_train=len(train_data))
            comparisons[f"HORIZON::{comp.context}"] = comp

    # Evaluar REGIME level
    if FallbackLevel.REGIME in calibrators:
        cal = calibrators[FallbackLevel.REGIME]
        regime_val: Dict[Tuple[str, str], List[Tuple[float, bool]]] = defaultdict(list)
        regime_test: Dict[Tuple[str, str], List[Tuple[float, bool]]] = defaultdict(list)
        
        for context, prob, outcome in val_data:
            regime_val[(context.strategy, context.regime)].append((prob, outcome))
        for context, prob, outcome in test_data:
            regime_test[(context.strategy, context.regime)].append((prob, outcome))
        
        for regime_key in regime_val:
            if regime_key not in regime_test:
                continue
            if len(regime_val[regime_key]) < min_val_samples:
                continue
            if len(regime_test[regime_key]) < min_test_samples:
                continue
            
            val_probs, val_outcomes = zip(*regime_val[regime_key])
            test_probs, test_outcomes = zip(*regime_test[regime_key])
            
            context = ContextKey(regime_key[0], 0, regime_key[1])
            comp = evaluate_context(cal, context, list(val_probs), list(val_outcomes),
                                    list(test_probs), list(test_outcomes),
                                    brier_tolerance, economic_tolerance)
            comp = CalibrationComparison(
                context=f"REGIME::{comp.context}",
                n_train=len(train_data),
                n_val=comp.n_val,
                n_test=comp.n_test,
                raw_brier=comp.raw_brier,
                raw_ece=comp.raw_ece,
                raw_log_loss=comp.raw_log_loss,
                raw_wilson_lb=comp.raw_wilson_lb,
                raw_economic_edge=comp.raw_economic_edge,
                calibrated_brier=comp.calibrated_brier,
                calibrated_ece=comp.calibrated_ece,
                calibrated_log_loss=comp.calibrated_log_loss,
                calibrated_wilson_lb=comp.calibrated_wilson_lb,
                calibrated_economic_edge=comp.calibrated_economic_edge,
                brier_improvement=comp.brier_improvement,
                ece_improvement=comp.ece_improvement,
                log_loss_improvement=comp.log_loss_improvement,
                wilson_improvement=comp.wilson_improvement,
                economic_impact=comp.economic_impact,
                verdict=comp.verdict,
                rejection_reason=comp.rejection_reason,
            )
            comparisons[comp.context] = comp
    
    # Evaluar STRATEGY level
    if FallbackLevel.STRATEGY in calibrators:
        cal = calibrators[FallbackLevel.STRATEGY]
        strat_val: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
        strat_test: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
        
        for context, prob, outcome in val_data:
            strat_val[context.strategy].append((prob, outcome))
        for context, prob, outcome in test_data:
            strat_test[context.strategy].append((prob, outcome))
        
        for strategy in strat_val:
            if strategy not in strat_test:
                continue
            if len(strat_val[strategy]) < min_val_samples:
                continue
            if len(strat_test[strategy]) < min_test_samples:
                continue
            
            val_probs, val_outcomes = zip(*strat_val[strategy])
            test_probs, test_outcomes = zip(*strat_test[strategy])
            
            context = ContextKey(strategy, 0, "ALL")
            comp = evaluate_context(cal, context, list(val_probs), list(val_outcomes),
                                    list(test_probs), list(test_outcomes),
                                    brier_tolerance, economic_tolerance)
            comp = CalibrationComparison(
                context=f"STRATEGY::{comp.context}",
                n_train=len(train_data),
                n_val=comp.n_val,
                n_test=comp.n_test,
                raw_brier=comp.raw_brier,
                raw_ece=comp.raw_ece,
                raw_log_loss=comp.raw_log_loss,
                raw_wilson_lb=comp.raw_wilson_lb,
                raw_economic_edge=comp.raw_economic_edge,
                calibrated_brier=comp.calibrated_brier,
                calibrated_ece=comp.calibrated_ece,
                calibrated_log_loss=comp.calibrated_log_loss,
                calibrated_wilson_lb=comp.calibrated_wilson_lb,
                calibrated_economic_edge=comp.calibrated_economic_edge,
                brier_improvement=comp.brier_improvement,
                ece_improvement=comp.ece_improvement,
                log_loss_improvement=comp.log_loss_improvement,
                wilson_improvement=comp.wilson_improvement,
                economic_impact=comp.economic_impact,
                verdict=comp.verdict,
                rejection_reason=comp.rejection_reason,
            )
            comparisons[comp.context] = comp
    
    # Evaluar GLOBAL level
    if FallbackLevel.GLOBAL in calibrators:
        cal = calibrators[FallbackLevel.GLOBAL]
        val_probs = [p for _, p, _ in val_data]
        val_outcomes = [o for _, _, o in val_data]
        test_probs = [p for _, p, _ in test_data]
        test_outcomes = [o for _, _, o in test_data]
        
        if len(val_probs) >= min_val_samples and len(test_probs) >= min_test_samples:
            context = ContextKey("GLOBAL", 0, "ALL")
            comp = evaluate_context(cal, context, val_probs, val_outcomes,
                                    test_probs, test_outcomes,
                                    brier_tolerance, economic_tolerance)
            comp = CalibrationComparison(
                context=f"GLOBAL::{comp.context}",
                n_train=len(train_data),
                n_val=comp.n_val,
                n_test=comp.n_test,
                raw_brier=comp.raw_brier,
                raw_ece=comp.raw_ece,
                raw_log_loss=comp.raw_log_loss,
                raw_wilson_lb=comp.raw_wilson_lb,
                raw_economic_edge=comp.raw_economic_edge,
                calibrated_brier=comp.calibrated_brier,
                calibrated_ece=comp.calibrated_ece,
                calibrated_log_loss=comp.calibrated_log_loss,
                calibrated_wilson_lb=comp.calibrated_wilson_lb,
                calibrated_economic_edge=comp.calibrated_economic_edge,
                brier_improvement=comp.brier_improvement,
                ece_improvement=comp.ece_improvement,
                log_loss_improvement=comp.log_loss_improvement,
                wilson_improvement=comp.wilson_improvement,
                economic_impact=comp.economic_impact,
                verdict=comp.verdict,
                rejection_reason=comp.rejection_reason,
            )
            comparisons[comp.context] = comp
    
    return comparisons