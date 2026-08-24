"""FASE 9.4 — Adaptive Expert Selection: la señal NO se diluye.

FASE 9.3 demostró que la señal de ZENIN vive en la SELECCIÓN (hard-max
del MoE: 68.4% NVDA, 60.0% AMD, 62.1% AAPL, 54.1% BTC) y que la mezcla
suave actual (~uniforme por el guardrail) la diluye a ~51-53%. Pero
pasar directamente a argmax → operar sería caer en otra trampa: el
ruido de selección amplificado.

9.4 introduce tres modos de selección controlada y, sobre todo, la
capacidad de decir NO TRADE:

    SOFT       — softmax sobre el score neto (lo que el evidence dice)
    SELECTIVE  — solo los mejores expertos con peso significativo
                 (umbral de calidad relativa + tope de expertos)
    HARD_MAX   — ganador único, SOLO si pasa guardrails
                 (muestra, historial, margen sobre el segundo, edge neto)

La escalera es defensiva: hard_max → selective → soft → hold. Cada
modo puede caer al siguiente si su guardrail falla. Y ANTES que todo:
si el mejor experto no tiene edge neto esperado (expected_return −
costo − penalidad de riesgo), la decisión es HOLD / NO TRADE.

El score de experto deja de ser accuracy (que ya demostró engañar:
Naive 62.2% en NVDA y aún pierde dinero) y pasa a:

    score = expected_net_return × calibration_quality × evidence_strength

    expected_net_return = expected_return − expected_cost − risk_penalty
    risk_penalty        = risk_aversion × desviación del PnL direccional
    calibration_quality = 1 − min(calibration_error, 1)
    evidence_strength   = min(n / min_n, 1)

La dirección (accuracy) queda como métrica secundaria/guardrail, no
como objetivo. Módulo puro, sin SQL (regla 3): consume ExpertScores
del PerformanceAnalyzer (FASE 8) y un CostModel (FASE 9.2).
"""

from __future__ import annotations

from .selection_types import (
    SelectionMode,
    SelectionConfig,
    ExpertNetScore,
    SelectionResult,
    DECISION_TRADE,
    DECISION_HOLD,
)
from .selection_compute import expert_net_scores, softmax, has_evidence

__all__ = [
    "SelectionMode",
    "SelectionConfig",
    "ExpertNetScore",
    "expert_net_scores",
    "SelectionResult",
    "select_weights",
]


def select_weights(
    net_scores: Sequence[ExpertNetScore],
    *,
    config: SelectionConfig,
) -> SelectionResult:
    """Pesos + decisión del contexto (experto × régimen × horizonte).

    Árbol de decisión (FASE 9.4):

        1. Sin expertos con muestra      -> HOLD (sin evidencia)
        2. Mejor edge neto <= umbral     -> HOLD / NO TRADE
        3. hard_max (con guardrails:
           muestra, historial, margen sobre el segundo)
           -> ganador único; si falla, cae a selective
        4. selective (con evidencia:
           calidad >= mejor × min_ratio, tope max_experts)
           -> re-pesaje softmax sobre los sobrevivientes;
           si falla, cae a soft
        5. soft                          -> softmax sobre todos

    Los pesos SIEMPRE suman 1 (portafolio totalmente invertido cuando
    hay decisión de trade).
    """
    ranked = sorted(net_scores, key=lambda s: s.score, reverse=True)
    if not ranked:
        return SelectionResult(
            config.mode, {}, None, DECISION_HOLD, "sin expertos con muestra en el contexto", ()
        )

    best = ranked[0]
    # ── puerta NO TRADE: sin edge neto no hay operación ──
    if best.expected_net <= config.min_expected_net:
        return SelectionResult(
            config.mode,
            {},
            None,
            DECISION_HOLD,
            (
                f"edge neto del mejor ({best.expert}) {best.expected_net:+.4f} "
                f"<= umbral {config.min_expected_net:+.4f} (expected {best.expected_return:+.4f} "
                f"- costo {best.expected_cost:.4f} - riesgo {best.risk_penalty:.4f})"
            ),
            tuple(ranked),
        )

    # ── hard_max: ganador único solo con guardrails ──
    if config.mode == SelectionMode.HARD_MAX:
        if has_evidence(best, config):
            second = ranked[1] if len(ranked) > 1 else None
            margin_ok = second is None or best.score >= second.score + config.min_margin
            if margin_ok:
                return SelectionResult(
                    SelectionMode.HARD_MAX,
                    {best.expert: 1.0},
                    best.expert,
                    DECISION_TRADE,
                    (
                        f"ganador único {best.expert} (score {best.score:+.4f}, "
                        f"n={best.n}) con guardrails"
                    ),
                    tuple(ranked),
                )
        # fallback controlado al modo selectivo

    # ── selective: solo los mejores con peso significativo ──
    if config.mode in (SelectionMode.SELECTIVE, SelectionMode.HARD_MAX):
        if has_evidence(best, config):
            survivors = [
                s
                for s in ranked
                if s.score >= best.score * config.min_ratio
            ][: config.max_experts]
            if survivors:
                return SelectionResult(
                    SelectionMode.SELECTIVE,
                    softmax(survivors, config.temperature),
                    best.expert,
                    DECISION_TRADE,
                    (
                        f"selectivo sobre {[s.expert for s in survivors]} "
                        f"(score >= {best.score * config.min_ratio:.4f}, "
                        f"máx {config.max_experts})"
                    ),
                    tuple(ranked),
                )
        # fallback controlado al modo suave

    # ── soft: softmax sobre todos los expertos con muestra ──
    return SelectionResult(
        SelectionMode.SOFT,
        softmax(ranked, config.temperature),
        best.expert,
        DECISION_TRADE,
        f"softmax sobre {[s.expert for s in ranked]} (temperature={config.temperature})",
        tuple(ranked),
    )