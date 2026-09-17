"""Predefined configuration profiles for Zephyr LiveBot."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iot_machine_learning.infrastructure.adapters.market.zephyr.config.bot_config import LiveBotConfig


def get_conservative_config() -> LiveBotConfig:
    """Configuración conservadora para capital real con aversión al riesgo."""
    from iot_machine_learning.infrastructure.adapters.market.zephyr.config.bot_config import LiveBotConfig

    return LiveBotConfig(
        max_position_pct=0.02,
        cooldown_ms=1000,
        phi_moe_threshold=0.6,
        emergency_lambda_threshold=0.9,
        max_lot_size=0.0005,
    )


def get_aggressive_config() -> LiveBotConfig:
    """Configuración agresiva para capturar momentum de alta frecuencia."""
    from iot_machine_learning.infrastructure.adapters.market.zephyr.config.bot_config import LiveBotConfig

    return LiveBotConfig(
        max_position_pct=0.1,
        cooldown_ms=100,
        phi_moe_threshold=0.4,
        emergency_lambda_threshold=0.95,
        max_lot_size=0.005,
    )


def get_testnet_config() -> LiveBotConfig:
    """Configuración estándar para pruebas en testnet o paper trading."""
    from iot_machine_learning.infrastructure.adapters.market.zephyr.config.bot_config import LiveBotConfig

    return LiveBotConfig(
        testnet=True,
        dry_run=False,
    )
