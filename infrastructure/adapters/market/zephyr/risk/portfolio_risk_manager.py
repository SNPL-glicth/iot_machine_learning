"""Portfolio Risk Manager — High-Water Mark profit preservation and correlation guardrails."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_CLUSTERS: dict[str, set[str]] = {
    "US_TECH_INDEX": {"SPY", "QQQ", "AAPL", "NVDA", "MSFT", "TSLA"},
}


@dataclass
class PortfolioRiskConfig:
    """Configuración para disyuntor de portafolio y control de correlación."""
    profit_lock_trigger_usd: float = 10.0   # Activa trinquete al superar este beneficio
    max_giveback_pct: float = 0.25          # Máximo retroceso permitido del pico (25%)
    max_cluster_positions: int = 1          # Máx posiciones simultáneas en misma dirección por cluster
    enable_macro_velocity_filter: bool = True
    macro_velocity_epsilon: float = 0.0001  # Tolerancia mínima para considerar velocidad no neutral
    max_daily_loss_usd: float = 50.0        # Límite de pérdida diaria absoluta (independiente de picos)


class PortfolioRiskManager:
    """Gestiona el riesgo agregado de la cuenta y evita degradación de beneficios acumulados."""

    def __init__(
        self,
        initial_equity: float,
        config: PortfolioRiskConfig | None = None,
        clusters: dict[str, set[str]] | None = None,
    ) -> None:
        self.config = config or PortfolioRiskConfig()
        self.initial_equity: float = initial_equity
        self.peak_equity: float = initial_equity
        self.is_profit_locked: bool = False
        self.circuit_breaker_tripped: bool = False
        self.clusters: dict[str, set[str]] = clusters or DEFAULT_CLUSTERS

    def reset_day(self, new_equity: float) -> None:
        """Reinicia el High-Water Mark diario para una nueva sesión de mercado."""
        self.initial_equity = new_equity
        self.peak_equity = new_equity
        self.is_profit_locked = False
        self.circuit_breaker_tripped = False
        logger.info(
            "Portfolio risk baseline reset for new market session: initial_equity=$%.2f",
            new_equity,
        )

    def update_equity(self, current_equity: float) -> tuple[bool, str]:
        """Evalúa el equity actual frente a la pérdida máxima absoluta y el High-Water Mark diario.

        Retorna:
            (tripped, reason): True si se dispara el disyuntor (por pérdida absoluta o por giveback de pico).
        """
        if self.circuit_breaker_tripped:
            return True, "Daily Circuit Breaker already tripped — all trading suspended"

        # 1. Comprobación de pérdida máxima diaria absoluta (independiente de si hubo ganancia previa)
        if self.config.max_daily_loss_usd > 0:
            daily_loss = self.initial_equity - current_equity
            if daily_loss >= abs(self.config.max_daily_loss_usd):
                self.circuit_breaker_tripped = True
                reason = (
                    f"CRITICAL: Maximum Daily Loss Limit Reached! "
                    f"Initial Equity: ${self.initial_equity:.2f}, Current Equity: ${current_equity:.2f}, "
                    f"Daily Loss: -${daily_loss:.2f} >= Limit: -${abs(self.config.max_daily_loss_usd):.2f}. "
                    "Suspending all trading for the rest of the day."
                )
                logger.critical(reason)
                return True, reason

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        peak_profit = self.peak_equity - self.initial_equity

        # 2. Comprobación de trinquete de ganancias (giveback desde el pico)
        if peak_profit >= self.config.profit_lock_trigger_usd:
            self.is_profit_locked = True
            max_giveback = peak_profit * self.config.max_giveback_pct
            floor_equity = self.peak_equity - max_giveback

            if current_equity <= floor_equity:
                self.circuit_breaker_tripped = True
                realized_pnl = current_equity - self.initial_equity
                reason = (
                    f"Portfolio Circuit Breaker Triggered: Peak profit was ${peak_profit:.2f}, "
                    f"equity fell to ${current_equity:.2f} <= floor ${floor_equity:.2f} "
                    f"(Preserved profit: ${realized_pnl:.2f})"
                )
                logger.warning(reason)
                return True, reason

        return False, ""

    def check_correlation_guardrail(
        self, symbol: str, side: str, open_positions: dict[str, float]
    ) -> tuple[bool, str]:
        """Impide abrir posiciones concurrentes en la misma dirección en activos altamente correlacionados."""
        sym = symbol.upper()
        target_cluster_name = None
        for name, members in self.clusters.items():
            if sym in members:
                target_cluster_name = name
                break

        if not target_cluster_name:
            return True, ""

        cluster_members = self.clusters[target_cluster_name]
        is_entering_long = side.lower() in ("buy", "long")

        same_side_count = 0
        conflicting_syms = []

        for other_sym, qty in open_positions.items():
            if other_sym.upper() in cluster_members and qty != 0.0:
                is_pos_long = qty > 0.0
                if is_pos_long == is_entering_long:
                    same_side_count += 1
                    conflicting_syms.append(f"{other_sym}({qty})")

        if same_side_count >= self.config.max_cluster_positions:
            reason = (
                f"Correlation Guardrail VETO: Cluster '{target_cluster_name}' already has "
                f"{same_side_count} {side.upper()} position(s): {', '.join(conflicting_syms)}"
            )
            logger.warning(reason)
            return False, reason

        return True, ""

    def check_macro_velocity(
        self, symbol: str, side: str, macro_velocity: float
    ) -> tuple[bool, str]:
        """Veta órdenes contrarias al impulso macroeconómico o aceleración del mercado."""
        if not self.config.enable_macro_velocity_filter:
            return True, ""

        eps = self.config.macro_velocity_epsilon
        side_lower = side.lower()

        if side_lower in ("sell", "short") and macro_velocity > eps:
            reason = (
                f"Macro Velocity VETO [{symbol}]: Cannot open SHORT while macro velocity is positive "
                f"({macro_velocity:+.4f} > {eps})"
            )
            logger.warning(reason)
            return False, reason

        if side_lower in ("buy", "long") and macro_velocity < -eps:
            reason = (
                f"Macro Velocity VETO [{symbol}]: Cannot open LONG while macro velocity is negative "
                f"({macro_velocity:+.4f} < -{eps})"
            )
            logger.warning(reason)
            return False, reason

        return True, ""

    def reset_daily(self, new_initial_equity: float) -> None:
        """Reinicia los umbrales al comienzo de una nueva sesión de mercado."""
        self.initial_equity = new_initial_equity
        self.peak_equity = new_initial_equity
        self.is_profit_locked = False
        self.circuit_breaker_tripped = False
