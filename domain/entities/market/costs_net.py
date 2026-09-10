"""Costos vivos y evaluación neta (FASE 6) — el edge después de pagar.

``costs.py`` (FASE 9.2) es la capa estática: perfiles por instrumento y
la escalera del edge. Este módulo es la capa viva: spread observado del
L1 (Fase 3), ``NetEvaluation`` (el número que manda al gate) y el
constructor de costos por símbolo que falla cerrado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .costs import (
    COST_PROFILES,
    CostModel,
    classify_edge,
    edge_ladder_index,
)

__all__ = [
    "NetEvaluation",
    "dynamic_cost_for",
    "evaluate_net",
    "with_observed_spread",
]


def with_observed_spread(
    base: CostModel, spread_bps_one_way: float
) -> CostModel:
    """Modelo con spread L1 observado sobre un perfil base.

    El spread L1 es instantáneo y one-way; el round-trip ≈ 2×
    observado. Slippage y comisión se heredan del perfil base.
    """
    if not isinstance(base, CostModel):
        raise TypeError("base debe ser CostModel")
    if not math.isfinite(spread_bps_one_way) or spread_bps_one_way < 0:
        raise ValueError(f"spread observado inválido: {spread_bps_one_way!r}")
    return CostModel(
        spread_bps=2.0 * spread_bps_one_way,
        slippage_bps=base.slippage_bps,
        commission_bps=base.commission_bps,
    )


def dynamic_cost_for(symbol: str, spread_bps_observed: float) -> CostModel:
    """Costo con spread L1 observado sobre el perfil del símbolo.

    Símbolo desconocido ⇒ ValueError (costos inventados son peores
    que ningún trade: el gate debe fallar cerrado, no abierto).
    """
    profile = COST_PROFILES.get(symbol)
    if profile is None:
        raise ValueError(
            f"sin perfil de costos para {symbol!r} "
            f"(conocidos: {sorted(COST_PROFILES)})"
        )
    return with_observed_spread(profile, spread_bps_observed)


@dataclass(frozen=True, slots=True, kw_only=True)
class NetEvaluation:
    """Edge después de pagar, con su clase de la escalera."""

    gross_return: float
    cost_fraction: float
    net_return: float
    edge: str  # clase de EDGE_LADDER

    def __post_init__(self) -> None:
        for name in ("gross_return", "cost_fraction", "net_return"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} no finito")
        if self.cost_fraction < 0:
            raise ValueError("cost_fraction no puede ser negativa")
        edge_ladder_index(self.edge)  # valida la clase


def evaluate_net(
    gross_return: float,
    cost_model: CostModel,
    *,
    sharpe: float | None = None,
    sharpe_threshold: float = 0.5,
) -> NetEvaluation:
    """Bruto − costos → neto + clase (el número que manda, FASE 6)."""
    if not isinstance(cost_model, CostModel):
        raise TypeError("cost_model debe ser CostModel")
    if not math.isfinite(gross_return):
        raise ValueError(f"gross_return no finito: {gross_return!r}")
    net = cost_model.net(gross_return)
    return NetEvaluation(
        gross_return=gross_return,
        cost_fraction=cost_model.total(),
        net_return=net,
        edge=classify_edge(gross_return, net, sharpe=sharpe,
                           sharpe_threshold=sharpe_threshold),
    )
