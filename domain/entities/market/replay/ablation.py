"""FASE 9.3 — Matriz de ablations: ¿de dónde salió el edge bruto?

FASE 9.2 demostró que ZENIN tiene señal bruta real (+0.03% a +0.07% en
acciones) pero insuficiente para superar costos (12–24 bps). La pregunta
de 9.3 es quirúrgica: **qué componente aporta ese edge bruto** y cuál
estorba. No se agrega nada: se REMUEVE un componente a la vez y se mide.

La matriz se evalúa sobre los MISMOS outcomes (predicciones persistidas
de las corridas reales): lo único que cambia entre ablaciones es el
vector de pesos aplicado a los expertos.

    Naive / Momentum / EMA crossover   — el experto solo (peso 1.0)
    ZENIN - memoria                    — pesos uniformes (sin adaptación)
    ZENIN - régimen                    — contexto global, sin dimensión régimen
    ZENIN - MoE                        — contexto + mejor experto único (hard max)
    ZENIN completo                     — la versión activa real (FASE 8/9.1)

Lectura honesta: si "ZENIN completo" ≤ baselines simples, ZENIN es un
montón de ingeniería para conseguir menos que una estrategia sencilla,
y la señal para 9.4 es SIMPLIFICAR, no meter complejidad.

Las versiones del modelo son una cadena append-only global (FASE 8); el
`reason` de cada versión registra "wf {symbol} W{index}: ...". Para cada
ventana se reconstruye la versión activa: la última creada por ese
símbolo con índice ≤ W; si no existe, la heredada al iniciar la corrida
(la inmediatamente anterior a la primera versión del símbolo); si el
símbolo nunca adaptó, la última versión global.
"""

from __future__ import annotations

from .ablation_constants import (
    ABLATION_NAIVE,
    ABLATION_MOMENTUM,
    ABLATION_EMA,
    ABLATION_NO_MEMORY,
    ABLATION_NO_REGIME,
    ABLATION_NO_MOE,
    ABLATION_FULL,
    ABLATIONS,
)
from .version_tracking import parse_version_reason, active_versions_by_window
from .ablation_weights import ablation_weights
from .portfolio import portfolio_net_returns, sharpe_of, max_drawdown
from .ablation_stats import (
    AblationWindow,
    ablation_window_stats,
    AblationStats,
    aggregate_ablation,
)
from .ablation_render import render_ablation_matrix

__all__ = [
    "ABLATION_NAIVE",
    "ABLATION_MOMENTUM",
    "ABLATION_EMA",
    "ABLATION_NO_MEMORY",
    "ABLATION_NO_REGIME",
    "ABLATION_NO_MOE",
    "ABLATION_FULL",
    "ABLATIONS",
    "parse_version_reason",
    "active_versions_by_window",
    "ablation_weights",
    "portfolio_net_returns",
    "AblationWindow",
    "ablation_window_stats",
    "AblationStats",
    "aggregate_ablation",
    "sharpe_of",
    "max_drawdown",
    "render_ablation_matrix",
]