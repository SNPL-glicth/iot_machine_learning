"""FASE 10.5 — Fallback Hierarchy Builder."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from .context_calibrator import ContextCalibrator, ContextKey, CalibrationMethod
from .verdicts import FallbackLevel


def build_fallback_calibrators(
    train_data: List[Tuple[ContextKey, float, bool]],
    method: CalibrationMethod,
    min_context_samples: int,
) -> Dict[FallbackLevel, ContextCalibrator]:
    """Construye calibradores para cada nivel de fallback."""
    
    calibrators: Dict[FallbackLevel, ContextCalibrator] = {}
    
    # Agrupar por contexto completo
    context_data: Dict[ContextKey, List[Tuple[float, bool]]] = defaultdict(list)
    for context, prob, outcome in train_data:
        context_data[context].append((prob, outcome))
    
    # 1. CONTEXT level: solo contextos con suficientes datos
    context_train = [(c, p, o) for c, items in context_data.items()
                    if len(items) >= min_context_samples
                    for p, o in items]
    if context_train:
        calibrators[FallbackLevel.CONTEXT] = _fit_calibrator(context_train, method)

    # 2. HORIZON level: strategy + horizon (el horizonte domina la tasa base)
    horizon_data: Dict[Tuple[str, int], List[Tuple[float, bool]]] = defaultdict(list)
    for context, prob, outcome in train_data:
        key = (context.strategy, context.horizon_seconds)
        horizon_data[key].append((prob, outcome))

    horizon_train = [(ContextKey(s, hz, "ALL"), p, o)
                     for (s, hz), items in horizon_data.items()
                     if len(items) >= min_context_samples
                     for p, o in items]
    if horizon_train:
        calibrators[FallbackLevel.HORIZON] = _fit_calibrator(horizon_train, method)

    # 3. REGIME level: strategy + regime
    regime_data: Dict[Tuple[str, str], List[Tuple[float, bool]]] = defaultdict(list)
    for context, prob, outcome in train_data:
        key = (context.strategy, context.regime)
        regime_data[key].append((prob, outcome))
    
    regime_train = [(ContextKey(s, 0, r), p, o) 
                   for (s, r), items in regime_data.items()
                   if len(items) >= min_context_samples
                   for p, o in items]
    if regime_train:
        calibrators[FallbackLevel.REGIME] = _fit_calibrator(regime_train, method)

    # 4. STRATEGY level: solo strategy
    strategy_data: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
    for context, prob, outcome in train_data:
        strategy_data[context.strategy].append((prob, outcome))
    
    strategy_train = [(ContextKey(s, 0, "ALL"), p, o) 
                     for s, items in strategy_data.items()
                     if len(items) >= min_context_samples
                     for p, o in items]
    if strategy_train:
        calibrators[FallbackLevel.STRATEGY] = _fit_calibrator(strategy_train, method)

    # 5. GLOBAL level: todos los datos reetiquetados bajo la clave canónica
    # GLOBAL (el lookup y la evaluación buscan esa clave; sin reetiquetado
    # fit() guardaría params bajo los contextos finos originales y el
    # calibrador global devolvería siempre identidad).
    if len(train_data) >= min_context_samples:
        global_train = [
            (ContextKey("GLOBAL", 0, "ALL"), p, o) for _c, p, o in train_data
        ]
        calibrators[FallbackLevel.GLOBAL] = _fit_calibrator(global_train, method)
    
    return calibrators


def _fit_calibrator(
    train_data: List[Tuple[ContextKey, float, bool]],
    method: CalibrationMethod,
) -> ContextCalibrator:
    """Entrena calibrador solo con datos de training."""
    calibrator = ContextCalibrator(method=method)
    calibrator.fit(train_data)
    return calibrator


def get_fallback_calibrator(
    context: ContextKey,
    available_calibrators: Dict[FallbackLevel, ContextCalibrator],
) -> Tuple[FallbackLevel, ContextCalibrator | None]:
    """Obtiene calibrador según fallback hierarchy."""
    
    # 1. Intentar context-specific
    if FallbackLevel.CONTEXT in available_calibrators:
        cal = available_calibrators[FallbackLevel.CONTEXT]
        if cal._params.get(context) and cal._params[context].is_valid:
            return FallbackLevel.CONTEXT, cal

    # 2. Intentar horizon-specific (strategy + horizon, sin regime)
    if FallbackLevel.HORIZON in available_calibrators:
        horizon_context = ContextKey(
            context.strategy, context.horizon_seconds, "ALL"
        )
        cal = available_calibrators[FallbackLevel.HORIZON]
        if cal._params.get(horizon_context) and cal._params[horizon_context].is_valid:
            return FallbackLevel.HORIZON, cal

    # 3. Intentar regime-specific
    if FallbackLevel.REGIME in available_calibrators:
        regime_context = ContextKey(context.strategy, 0, context.regime)
        cal = available_calibrators[FallbackLevel.REGIME]
        if cal._params.get(regime_context) and cal._params[regime_context].is_valid:
            return FallbackLevel.REGIME, cal
    
    # 4. Intentar strategy-specific
    if FallbackLevel.STRATEGY in available_calibrators:
        strategy_context = ContextKey(context.strategy, 0, "ALL")
        cal = available_calibrators[FallbackLevel.STRATEGY]
        if cal._params.get(strategy_context) and cal._params[strategy_context].is_valid:
            return FallbackLevel.STRATEGY, cal
    
    # 5. Intentar global
    if FallbackLevel.GLOBAL in available_calibrators:
        global_context = ContextKey("GLOBAL", 0, "ALL")
        cal = available_calibrators[FallbackLevel.GLOBAL]
        if cal._params.get(global_context) and cal._params[global_context].is_valid:
            return FallbackLevel.GLOBAL, cal
    
    # 6. Sin calibración disponible
    return FallbackLevel.UNAVAILABLE, None

def resolve_fallback_context(
    context: ContextKey, level: FallbackLevel
) -> ContextKey:
    """Clave bajo la que el calibrador del nivel seleccionado ajustó params.

    El lookup de ``get_fallback_calibrator`` valida existencia contra la
    clave gruesa del nivel (p.ej. ``ContextKey(strategy, 0, "ALL")`` en
    STRATEGY); ``ContextCalibrator.calibrate`` busca con la clave que recibe.
    Pasarle el contexto fino original hacía miss garantizado en todos los
    niveles salvo CONTEXT: los calibradores se seleccionaban pero nunca
    aplicaban su transformación. Este resolvedor es el contrato único entre
    selección y aplicación.
    """
    if level == FallbackLevel.CONTEXT:
        return context
    if level == FallbackLevel.HORIZON:
        return ContextKey(context.strategy, context.horizon_seconds, "ALL")
    if level == FallbackLevel.REGIME:
        return ContextKey(context.strategy, 0, context.regime)
    if level == FallbackLevel.STRATEGY:
        return ContextKey(context.strategy, 0, "ALL")
    return ContextKey("GLOBAL", 0, "ALL")
