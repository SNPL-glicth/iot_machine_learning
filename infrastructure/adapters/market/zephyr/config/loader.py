"""Configuration loader for ZENIN & Zephyr trading platform.

Provides canonical loading routines for public JSON configurations and local secrets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, cast

from iot_machine_learning.infrastructure.adapters.market.zephyr.config.bot_config import (
    LiveBotConfig,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.config.config_manager import (
    ConfigManager,
)

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent
ZEPHYR_CONFIG_PATH = CONFIG_DIR / "zephyr_config.json"
ZEPHYR_SECRETS_PATH = CONFIG_DIR / "zephyr_secrets.json"
ZEPHYR_SECRETS_TEMPLATE_PATH = CONFIG_DIR / "zephyr_secrets.json.template"
ALPACA_PAPER_CONFIG_PATH = CONFIG_DIR / "live_alpaca_paper.json"


def load_zephyr_config(
    config_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
    profile_name: Optional[str] = None,
) -> LiveBotConfig:
    """Loads Zephyr configuration from JSON merged with local secrets, profiles, and env vars.

    Args:
        config_path: Path to public configuration JSON. Defaults to zephyr_config.json.
        secrets_path: Path to secrets JSON file. Defaults to zephyr_secrets.json.
        profile_name: Optional profile name to load from AccountManager.

    Returns:
        LiveBotConfig: Fully validated configuration instance.
    """
    cfg_file = Path(config_path) if config_path else ZEPHYR_CONFIG_PATH
    sec_file = Path(secrets_path) if secrets_path else ZEPHYR_SECRETS_PATH

    manager = ConfigManager(config_path=cfg_file, secrets_path=sec_file)
    return manager.load_config(profile_name=profile_name)


def get_available_configs() -> list[str]:
    """Lists all available JSON configuration profiles in the config directory.

    Returns:
        list[str]: List of filenames for available configurations.
    """
    return [p.name for p in CONFIG_DIR.glob("*.json") if not p.name.endswith("secrets.json")]


__all__ = [
    "CONFIG_DIR",
    "ZEPHYR_CONFIG_PATH",
    "ZEPHYR_SECRETS_PATH",
    "ZEPHYR_SECRETS_TEMPLATE_PATH",
    "ALPACA_PAPER_CONFIG_PATH",
    "load_zephyr_config",
    "get_available_configs",
]
