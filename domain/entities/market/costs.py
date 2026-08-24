"""Costos reales de ejecución (FASE 9.2) — el edge después de pagar.

Hasta FASE 9.1 preguntamos "¿predijo correctamente?". FASE 9.2 pregunta
"¿después de pagar por intentarlo, todavía ganó?". La respuesta honesta
se construye por predicción:

    expected_return            (el edge bruto que el modelo declara)
    expected_cost              (spread + slippage + comisión, por instrumento)
    expected_net_return        (el edge después de costos)

Y el EDGE se clasifica en una escalera:

    gross_negative             (ni siquiera en bruto hay edge)
    cost_negative              (hay señal bruta, pero los costos la matan)
    risk_negative              (neto > 0 pero sin consistencia: sharpe bajo)
    cost_positive              (neto > 0, sin ajuste de riesgo disponible)
    risk_adjusted_positive     (neto > 0 Y sharpe >= umbral)

Un resultado ``cost_negative`` no es un fracaso: es un descubrimiento
brutalmente útil (capacidad predictiva real insuficiente para superar
los costos del mercado).

Supuestos (documentados y auditables, retail, orden de mercado, ida y
vuelta redondeada a bps):

- Acciones (NVDA, AAPL, AMD, MSFT, QQQ, SPY): spread 4 bps + slippage
  5 bps + comisión 3 bps = 12 bps (ejemplo de la FASE 9.2).
- Cripto (BTC-USD, ETH-USD): spread 2 bps + slippage 2 bps + comisión
  20 bps (taker) = 24 bps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "CostModel",
    "COST_PROFILES",
    "DEFAULT_STOCK_COSTS",
    "classify_edge",
    "edge_ladder_index",
]

# Escalera del EDGE: índice más alto = mejor clase alcanzada.
EDGE_GROSS_NEGATIVE: Final = "gross_negative"
EDGE_COST_NEGATIVE: Final = "cost_negative"
EDGE_RISK_NEGATIVE: Final = "risk_negative"
EDGE_COST_POSITIVE: Final = "cost_positive"
EDGE_RISK_ADJUSTED_POSITIVE: Final = "risk_adjusted_positive"

EDGE_LADDER: Final = (
    EDGE_GROSS_NEGATIVE,
    EDGE_COST_NEGATIVE,
    EDGE_RISK_NEGATIVE,
    EDGE_COST_POSITIVE,
    EDGE_RISK_ADJUSTED_POSITIVE,
)


def edge_ladder_index(edge: str) -> int:
    """Índice de la clase en la escalera (para ordenar agregados)."""
    try:
        return EDGE_LADDER.index(edge)
    except ValueError:
        raise ValueError(f"clase de edge desconocida: {edge!r}") from None


@dataclass(frozen=True, slots=True, kw_only=True)
class CostModel:
    """Modelo de costos por instrumento, en bps (0.01% = 1 bp).

    Attributes:
        spread_bps: Costo de cruzar el spread (ida y vuelta, redondeado).
        slippage_bps: Deslizamiento esperado (ida y vuelta).
        commission_bps: Comisión del broker (ida y vuelta).
    """

    spread_bps: float = 4.0
    slippage_bps: float = 5.0
    commission_bps: float = 3.0

    def __post_init__(self) -> None:
        for name in ("spread_bps", "slippage_bps", "commission_bps"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1000.0:
                raise ValueError(f"{name} fuera de rango [0, 1000] bps: {value!r}")

    @property
    def total_bps(self) -> float:
        """Costo total redondeado a bps (fracción de 0.0001)."""
        return round(self.spread_bps + self.slippage_bps + self.commission_bps)

    def total(self) -> float:
        """Costo total como fracción (0.0012 = 12 bps = 0.12%)."""
        return self.total_bps / 10000.0

    def net(self, gross_return: float) -> float:
        """Retorno neto = bruto − costo total (fracciones, 0.0012 = 0.12%)."""
        return gross_return - self.total()

    def __str__(self) -> str:
        return (
            f"{self.total_bps}bps "
            f"(spread {self.spread_bps:.0f} + slippage {self.slippage_bps:.0f} "
            f"+ comisión {self.commission_bps:.0f})"
        )


# Perfiles por instrumento (ver docstring del módulo).
DEFAULT_STOCK_COSTS: Final = CostModel(spread_bps=4.0, slippage_bps=5.0, commission_bps=3.0)
_CRYPTO_COSTS: Final = CostModel(spread_bps=2.0, slippage_bps=2.0, commission_bps=20.0)

COST_PROFILES: Final = {
    "NVDA": DEFAULT_STOCK_COSTS,
    "AAPL": DEFAULT_STOCK_COSTS,
    "AMD": DEFAULT_STOCK_COSTS,
    "MSFT": DEFAULT_STOCK_COSTS,
    "QQQ": DEFAULT_STOCK_COSTS,
    "SPY": DEFAULT_STOCK_COSTS,
    "BTC-USD": _CRYPTO_COSTS,
    "ETH-USD": _CRYPTO_COSTS,
}


def classify_edge(
    gross_return: float,
    net_return: float,
    *,
    sharpe: float | None = None,
    sharpe_threshold: float = 0.5,
) -> str:
    """Clasifica el edge (bruto, neto, ajustado por riesgo) en la escalera.

    La clase devuelta es la MÁS ALTA alcanzada:

    - bruto <= 0            -> gross_negative (no hay edge ni en bruto);
    - neto <= 0             -> cost_negative (la señal murió a los costos);
    - sharpe < umbral       -> risk_negative (positivo pero inconsistente);
    - sharpe es None        -> cost_positive (neto > 0 sin ajuste);
    - si no                 -> risk_adjusted_positive.
    """
    if not gross_return > 0:
        return EDGE_GROSS_NEGATIVE
    if not net_return > 0:
        return EDGE_COST_NEGATIVE
    if sharpe is not None:
        if sharpe < sharpe_threshold:
            return EDGE_RISK_NEGATIVE
        return EDGE_RISK_ADJUSTED_POSITIVE
    return EDGE_COST_POSITIVE
