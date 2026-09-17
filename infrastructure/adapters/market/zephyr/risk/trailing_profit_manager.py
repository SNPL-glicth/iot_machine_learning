"""Trailing Profit Manager — Dynamic High-Water Mark profit protection without fixed caps."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrailingProfitConfig:
    """Configuración para la protección de ganancias por High-Water Mark."""
    activation_pnl_usd: float = 4.50      # Ganancia mínima para armar el trailing (permite expandir la operación)
    giveback_ratio: float = 0.30         # Máximo porcentaje de retroceso desde el pico (30%)
    min_giveback_usd: float = 1.80       # Retroceso mínimo en dólares para disparar salida (~20-25c oscilación)
    max_giveback_usd: float = 3.50       # Tope máximo de retroceso permitido


class TrailingProfitManager:
    """Monitorea el pico máximo de ganancia no realizada y dispara cierre si retrocede."""

    def __init__(self, config: TrailingProfitConfig | None = None) -> None:
        self.config = config or TrailingProfitConfig()
        self.peak_pnl: float = 0.0
        self.is_active: bool = False

    def update(self, unrealized_pnl: float) -> tuple[bool, str]:
        """
        Evalúa el PnL no realizado actual contra el High-Water Mark.

        Retorna:
            (should_exit, reason): True si el retroceso desde el pico superó la tolerancia.
        """
        # Si aún no está en ganancia significativa, no activa protección
        if unrealized_pnl <= 0.0:
            if not self.is_active:
                self.peak_pnl = 0.0
            return False, ""

        # Actualizar el pico más alto alcanzado (sin límite / sin techo)
        if unrealized_pnl > self.peak_pnl:
            self.peak_pnl = unrealized_pnl

        # Si el pico superó el umbral de activación, activamos el trailing lock
        if self.peak_pnl >= self.config.activation_pnl_usd:
            self.is_active = True
            giveback = max(
                self.config.min_giveback_usd,
                min(self.peak_pnl * self.config.giveback_ratio, self.config.max_giveback_usd),
            )
            threshold = self.peak_pnl - giveback
            if unrealized_pnl <= threshold:
                reason = (
                    f"Trailing Profit Lock: Peak was ${self.peak_pnl:.2f}, "
                    f"current is ${unrealized_pnl:.2f} (retraction > ${giveback:.2f})"
                )
                return True, reason

        return False, ""

    def reset(self) -> None:
        """Reinicia el estado al cerrar o aplanar la posición."""
        self.peak_pnl = 0.0
        self.is_active = False
