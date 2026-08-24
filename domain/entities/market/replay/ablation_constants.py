"""Ablation constants for FASE 9.3."""

from __future__ import annotations

from typing import Final

ABLATION_NAIVE = "Naive"
ABLATION_MOMENTUM = "Momentum"
ABLATION_EMA = "EMA crossover"
ABLATION_NO_MEMORY = "ZENIN - memoria"
ABLATION_NO_REGIME = "ZENIN - régimen"
ABLATION_NO_MOE = "ZENIN - MoE"
ABLATION_FULL = "ZENIN completo"

ABLATIONS: Final = (
    ABLATION_NAIVE,
    ABLATION_MOMENTUM,
    ABLATION_EMA,
    ABLATION_NO_MEMORY,
    ABLATION_NO_REGIME,
    ABLATION_NO_MOE,
    ABLATION_FULL,
)

_BASELINE_EXPERTS: Final = {
    ABLATION_NAIVE: "naive",
    ABLATION_MOMENTUM: "momentum",
    ABLATION_EMA: "ema-crossover",
}