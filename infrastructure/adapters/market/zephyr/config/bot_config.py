"""LiveBotConfig -- Configuración canónica institucional del bot live event-driven."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from iot_machine_learning.infrastructure.adapters.market.zephyr.config.presets import (
    get_aggressive_config,
    get_conservative_config,
    get_testnet_config,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.config.validator import (
    ConfigSerializationMixin,
    validate_live_bot_config,
)


@dataclass
class LiveBotConfig(ConfigSerializationMixin):
    """Configuración institucional completa del bot live event-driven."""

    broker: Literal["binance", "alpaca"] = "binance"
    symbol: str = "BTCUSDT"
    symbols: list[str] = field(default_factory=list)
    exchange: str = "binance"
    testnet: bool = True

    # Alpaca credentials & endpoints
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    alpaca_api_base_url: str = "https://paper-api.alpaca.markets/v2"
    alpaca_data_feed: str = "iex"
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str | None = None

    # Data streams
    depth_speed: str = "100ms"
    include_trades: bool = True
    include_book_ticker: bool = True
    include_kline: bool = False
    kline_interval: str = "1m"

    # Master Equation & Rosa Roja sovereignty
    rosa_roja_enabled: bool = True
    rosa_roja_min_history: int = 50
    use_master_orchestrator: bool = True
    master_shadow_mode: bool = False
    tau_mom: float = 0.5
    sigma_mom: float = 0.001

    # Risk & Execution parameters
    max_position_pct: float = 0.05
    max_concurrent_positions: int = 2
    max_portfolio_exposure_pct: float = 0.15
    max_lot_size: float = 0.001
    min_lot_size: float = 0.00001
    lot_size: float = 0.00001

    # Cooldown & Hysteresis
    cooldown_ms: int = 500
    min_price_change_pct: float = 0.0002
    dynamic_cooldown: bool = True

    # Master Equation thresholds
    phi_moe_threshold: float = 0.5
    geometric_threshold: float = -0.1
    emergency_lambda_threshold: float = 0.95

    # Order types
    use_post_only: bool = True
    market_on_high_accel: bool = True
    high_accel_threshold: float = 0.8

    # Stop Loss & Trailing Profit
    default_stop_pct: float = 0.02
    max_trade_loss_usd: float = 10.00
    default_target_pct: float = 0.04
    use_trailing_profit: bool = True
    trailing_activation_pnl: float = 15.00
    trailing_giveback_ratio: float = 0.25
    trailing_min_giveback: float = 5.00
    trailing_max_giveback: float = 12.00

    # Circuit breakers
    max_daily_loss_usd: float = 50.0
    portfolio_profit_lock_trigger: float = 12.0
    portfolio_max_giveback_pct: float = 0.25
    enforce_portfolio_profit_lock: bool = True

    # Loss streaks & clusters
    max_consecutive_losses: int = 2
    consecutive_loss_cooldown_sec: float = 900.0
    max_cluster_correlated_positions: int = 2
    asset_clusters: dict[str, set[str]] | None = None

    # Macro filter & session
    enforce_macro_velocity_alignment: bool = True
    enforce_market_hours: bool = True

    # Emergency Flush
    emergency_cancel_all: bool = True
    emergency_close_position: bool = True

    # Connectivity
    ws_reconnect_base_delay: float = 1.0
    ws_reconnect_max_delay: float = 60.0
    ws_ping_interval: float = 20.0
    ws_ping_timeout: float = 10.0
    ws_max_queue_size: int = 10000

    # Order Book
    ob_max_levels: int = 100
    ob_snapshot_interval_sec: float = 30.0

    # Persistence & Audit
    audit_log_path: str | None = "./logs/audit"
    audit_rotate_daily: bool = True
    state_snapshot_path: str | None = "./data/state_snapshots"
    snapshot_interval_sec: int = 300

    # Health checks & Feature flags
    health_check_interval_sec: int = 10
    max_latency_ms: float = 25.0
    enable_audit_log: bool = True
    enable_metrics_export: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        validate_live_bot_config(self)

    @property
    def is_paper_trading(self) -> bool:
        if self.broker == "alpaca":
            return "paper-api.alpaca.markets" in self.alpaca_api_base_url
        if self.broker == "binance":
            return self.testnet
        return False

    @property
    def is_live_trading(self) -> bool:
        return not self.is_paper_trading

    @property
    def dynamic_cooldown_ms(self) -> float:
        return float(self.cooldown_ms)

    def get_effective_cooldown(self, lambda_t: float) -> float:
        """Calcula cooldown dinámico según lambda_t."""
        if not self.dynamic_cooldown:
            return float(self.cooldown_ms)
        base = float(self.cooldown_ms)
        if lambda_t >= 0.8:
            return max(base * 0.2, 100.0)
        elif lambda_t >= 0.5:
            return base * 0.5
        return base


__all__ = [
    "LiveBotConfig",
    "get_conservative_config",
    "get_aggressive_config",
    "get_testnet_config",
]
