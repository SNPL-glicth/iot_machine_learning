"""LiveBotConfig -- Configuración del bot live event-driven."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class LiveBotConfig:
    """Configuración completa del bot live.

    Todos los parámetros son explícitos y validados en __post_init__.
    """

    # Broker selection
    broker: Literal["binance", "alpaca"] = "binance"

    # Símbolo e intercambio
    symbol: str = "BTCUSDT"
    symbols: list[str] = field(default_factory=list)
    exchange: str = "binance"
    testnet: bool = True

    # Alpaca-specific config
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    alpaca_api_base_url: str = "https://paper-api.alpaca.markets/v2"
    alpaca_data_feed: str = "iex"  # "iex" or "sip"

    # Streams de datos
    depth_speed: str = "100ms"  # "100ms" o "1000ms"
    include_trades: bool = True
    include_book_ticker: bool = True
    include_kline: bool = False
    kline_interval: str = "1m"

    # Motor Rosa Roja
    rosa_roja_enabled: bool = True
    rosa_roja_min_history: int = 50

    # Parámetros de riesgo y ejecución
    max_position_pct: float = 0.05      # 5% del equity máximo por trade
    max_concurrent_positions: int = 2   # Máximo de posiciones simultáneas en el portafolio
    max_portfolio_exposure_pct: float = 0.15  # Exposición combinada máxima (15%)
    max_lot_size: float = 0.001         # Tamaño máximo de lote (BTC)
    min_lot_size: float = 0.00001       # Tamaño mínimo de lote
    lot_size: float = 0.00001           # Incremento de lote (step size)

    # Cooldown / Histeresis
    cooldown_ms: int = 500              # Tiempo mínimo entre órdenes
    min_price_change_pct: float = 0.0002  # 0.02% cambio mínimo de precio
    dynamic_cooldown: bool = True       # Cooldown dinámico según lambda_t

    # Umbrales de decisión
    phi_moe_threshold: float = 0.5      # Phi_MoE >= 0.5 -> EXECUTE
    geometric_threshold: float = -0.1   # cos(theta) < -0.1 -> EMERGENCY_FLUSH
    emergency_lambda_threshold: float = 0.95  # lambda_t > 0.95 -> EMERGENCY_FLUSH

    # Tipos de orden
    use_post_only: bool = True          # LIMIT Post-Only (GTX) para EXECUTE normal
    market_on_high_accel: bool = True   # MARKET si aceleración alta
    high_accel_threshold: float = 0.8   # lambda_t > 0.8 -> alta aceleración

    # Stop Loss / Take Profit / Trailing High-Water Mark
    default_stop_pct: float = 0.02      # 2% stop loss
    max_trade_loss_usd: float = 10.00   # Stop loss máximo por trade en USD (software cut)
    default_target_pct: float = 0.04    # 4% take profit
    use_trailing_profit: bool = True    # Trailing High-Water Mark dinámico
    trailing_activation_pnl: float = 15.00 # PnL USD para activar trailing (~0.15% en $10k pos)
    trailing_giveback_ratio: float = 0.25 # Retroceso permitido (25%)
    trailing_min_giveback: float = 5.00   # Colchón mínimo de retroceso USD (~$0.17 en 30sh)
    trailing_max_giveback: float = 12.00  # Colchón máximo de retroceso USD

    # Protección de Pérdida Diaria y Ganancias (Circuit Breakers a nivel cuenta)
    max_daily_loss_usd: float = 50.0              # Límite de pérdida diaria máxima absoluta (ej. -$50 USD)
    portfolio_profit_lock_trigger: float = 30.0   # Ganancia pico USD para activar bloqueo
    portfolio_max_giveback_pct: float = 0.25      # Máximo retroceso permitido del pico (25%)
    enforce_portfolio_profit_lock: bool = True    # Circuit breaker diario activo

    # Control de Rachas de Pérdidas Consecutivas
    max_consecutive_losses: int = 2               # Racha máxima de pérdidas antes de pausa
    consecutive_loss_cooldown_sec: float = 900.0  # Tiempo de enfriamiento tras racha (15 minutos = 900s)

    # Guardrail de Correlación Cruzada (Cluster Tech)
    max_cluster_correlated_positions: int = 1     # Máx posiciones simultáneas en misma dirección en cluster Tech

    # Filtro de Inercia y Velocidad Macro
    enforce_macro_velocity_alignment: bool = True # Prohíbe cortos si velocidad macro > 0

    # Sesión de mercado
    enforce_market_hours: bool = True   # Valida sesión RTH 16:00 ET en Alpaca (inhibe fuera de RTH y aplana al cierre)

    # Emergency Flush
    emergency_cancel_all: bool = True   # Cancelar todas las órdenes abiertas
    emergency_close_position: bool = True  # Cerrar posición a mercado

    # Conectividad
    ws_reconnect_base_delay: float = 1.0
    ws_reconnect_max_delay: float = 60.0
    ws_ping_interval: float = 20.0
    ws_ping_timeout: float = 10.0
    ws_max_queue_size: int = 10000

    # Order Book
    ob_max_levels: int = 100
    ob_snapshot_interval_sec: float = 30.0

    # Persistencia y auditoría
    audit_log_path: str | None = "./logs/audit"
    audit_rotate_daily: bool = True
    state_snapshot_path: str | None = "./data/state_snapshots"
    snapshot_interval_sec: int = 300    # 5 minutos

    # Health checks
    health_check_interval_sec: int = 10
    max_latency_ms: float = 5.0         # Alerta si latencia feed->order > 5ms

    # Feature flags
    enable_audit_log: bool = True
    enable_metrics_export: bool = False
    dry_run: bool = False               # Solo simula, no envía órdenes reales

    def __post_init__(self) -> None:
        if not self.symbols:
            self.symbols = [self.symbol.upper()]
        else:
            self.symbols = [s.upper() for s in self.symbols]
            if self.symbol.upper() not in self.symbols:
                self.symbol = self.symbols[0]

        # Validaciones
        if self.max_position_pct <= 0 or self.max_position_pct > 1:
            raise ValueError("max_position_pct debe estar en (0, 1]")
        if self.cooldown_ms < 0:
            raise ValueError("cooldown_ms debe ser >= 0")
        if self.min_price_change_pct < 0:
            raise ValueError("min_price_change_pct debe ser >= 0")
        if not 0 <= self.phi_moe_threshold <= 1:
            raise ValueError("phi_moe_threshold debe estar en [0, 1]")
        if not -1 <= self.geometric_threshold <= 1:
            raise ValueError("geometric_threshold debe estar en [-1, 1]")
        if not 0 <= self.emergency_lambda_threshold <= 1:
            raise ValueError("emergency_lambda_threshold debe estar en [0, 1]")
        if self.lot_size <= 0:
            raise ValueError("lot_size debe ser > 0")
        if self.max_lot_size < self.min_lot_size:
            raise ValueError("max_lot_size debe ser >= min_lot_size")

        # Validación Alpaca
        if self.broker == "alpaca":
            if not self.alpaca_api_key:
                self.alpaca_api_key = os.getenv("ALPACA_API_KEY") or None
            if not self.alpaca_secret_key:
                self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY") or None
            if not self.alpaca_api_base_url:
                self.alpaca_api_base_url = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets/v2")

            if not self.alpaca_api_key:
                raise ValueError("alpaca_api_key es requerido cuando broker=alpaca")
            if not self.alpaca_secret_key:
                raise ValueError("alpaca_secret_key es requerido cuando broker=alpaca")
            if not self.alpaca_api_base_url:
                raise ValueError("alpaca_api_base_url es requerido cuando broker=alpaca")
            if "paper-api.alpaca.markets" not in self.alpaca_api_base_url:
                raise ValueError("alpaca_api_base_url debe ser el endpoint PAPER: https://paper-api.alpaca.markets")
            if self.alpaca_data_feed not in ("iex", "sip"):
                raise ValueError("alpaca_data_feed debe ser 'iex' o 'sip'")

    @property
    def is_paper_trading(self) -> bool:
        """Verifica si la configuración corresponde a paper trading."""
        if self.broker == "alpaca":
            return "paper-api.alpaca.markets" in self.alpaca_api_base_url
        if self.broker == "binance":
            return self.testnet
        return False

    @property
    def is_live_trading(self) -> bool:
        """Verifica si la configuración corresponde a live trading."""
        return not self.is_paper_trading

    def to_json(self) -> str:
        """Serializa a JSON."""
        return json.dumps({
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }, indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> LiveBotConfig:
        """Crea config desde JSON."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_file(cls, path: Path) -> LiveBotConfig:
        """Carga config desde archivo."""
        return cls.from_json(path.read_text())

    def save_to_file(self, path: Path) -> None:
        """Guarda config a archivo."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @property
    def dynamic_cooldown_ms(self) -> float:
        """Cooldown efectivo (puede ser sobrescrito en runtime)."""
        return float(self.cooldown_ms)

    def get_effective_cooldown(self, lambda_t: float) -> float:
        """Calcula cooldown dinámico basado en lambda_t.

        - lambda_t < 0.5: cooldown normal
        - 0.5 <= lambda_t < 0.8: cooldown * 0.5
        - lambda_t >= 0.8: cooldown * 0.2 (100ms mínimo)
        """
        if not self.dynamic_cooldown:
            return float(self.cooldown_ms)

        base = float(self.cooldown_ms)
        if lambda_t >= 0.8:
            return max(base * 0.2, 100.0)
        elif lambda_t >= 0.5:
            return base * 0.5
        return base


# Configuraciones predefinidas
def get_conservative_config() -> LiveBotConfig:
    """Configuración conservadora para capital real."""
    return LiveBotConfig(
        max_position_pct=0.02,
        cooldown_ms=1000,
        phi_moe_threshold=0.6,
        emergency_lambda_threshold=0.9,
        max_lot_size=0.0005,
    )


def get_aggressive_config() -> LiveBotConfig:
    """Configuración agresiva para testing."""
    return LiveBotConfig(
        max_position_pct=0.1,
        cooldown_ms=100,
        phi_moe_threshold=0.4,
        emergency_lambda_threshold=0.95,
        max_lot_size=0.005,
    )


def get_testnet_config() -> LiveBotConfig:
    """Configuración por defecto para testnet."""
    return LiveBotConfig(
        testnet=True,
        dry_run=True,
    )
