"""Validation and secure serialization logic for trading bot configurations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def validate_live_bot_config(config: Any) -> None:
    """Valida los parámetros numéricos y credenciales de broker de LiveBotConfig.

    Args:
        config: Objeto LiveBotConfig a validar.

    Raises:
        ValueError: Si algún parámetro se encuentra fuera de los límites permitidos.
    """
    if not config.symbols:
        config.symbols = [config.symbol.upper()]
    else:
        config.symbols = [s.upper() for s in config.symbols]
        if config.symbol.upper() not in config.symbols:
            config.symbol = config.symbols[0]

    if config.max_position_pct <= 0 or config.max_position_pct > 1:
        raise ValueError("max_position_pct debe estar en (0, 1]")
    if config.cooldown_ms < 0:
        raise ValueError("cooldown_ms debe ser >= 0")
    if config.min_price_change_pct < 0:
        raise ValueError("min_price_change_pct debe ser >= 0")
    if not 0 <= config.phi_moe_threshold <= 1:
        raise ValueError("phi_moe_threshold debe estar en [0, 1]")
    if not -1 <= config.geometric_threshold <= 1:
        raise ValueError("geometric_threshold debe estar en [-1, 1]")
    if not 0 <= config.emergency_lambda_threshold <= 1:
        raise ValueError("emergency_lambda_threshold debe estar en [0, 1]")
    if config.lot_size <= 0:
        raise ValueError("lot_size debe ser > 0")
    if config.max_lot_size < config.min_lot_size:
        raise ValueError("max_lot_size debe ser >= min_lot_size")

    if config.broker == "alpaca":
        if not config.alpaca_api_key:
            config.alpaca_api_key = os.getenv("ALPACA_API_KEY") or None
        if not config.alpaca_secret_key:
            config.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY") or None
        if not config.alpaca_api_base_url:
            config.alpaca_api_base_url = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets/v2")

        if not config.alpaca_api_key:
            raise ValueError("alpaca_api_key es requerido cuando broker=alpaca")
        if not config.alpaca_secret_key:
            raise ValueError("alpaca_secret_key es requerido cuando broker=alpaca")
        if not config.alpaca_api_base_url:
            raise ValueError("alpaca_api_base_url es requerido cuando broker=alpaca")
        if "paper-api.alpaca.markets" not in config.alpaca_api_base_url:
            raise ValueError("alpaca_api_base_url debe ser el endpoint PAPER: https://paper-api.alpaca.markets")
        if config.alpaca_data_feed not in ("iex", "sip"):
            raise ValueError("alpaca_data_feed debe ser 'iex' o 'sip'")


class ConfigSerializationMixin:
    """Mixin para serialización y persistencia segura de configuración."""

    SECRET_KEYS = (
        "alpaca_api_key",
        "alpaca_secret_key",
        "binance_api_key",
        "binance_api_secret",
        "weaviate_api_key",
    )

    def to_safe_dict(self) -> dict[str, Any]:
        """Extrae diccionario sanitizado sustituyendo secretos por redaction tokens."""
        return {
            k: ("***REDACTED***" if k in self.SECRET_KEYS and v else v)
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def to_public_dict(self) -> dict[str, Any]:
        """Extrae diccionario libre de claves secretas para guardado seguro en disco."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and k not in self.SECRET_KEYS
        }

    def to_json(self) -> str:
        """Serializa a formato JSON legible."""
        return json.dumps({
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }, indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> Any:
        return cls(**json.loads(json_str))

    @classmethod
    def from_file(cls, path: Any) -> Any:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save_to_file(self, path: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load_with_secrets(
        cls,
        config_path: Any | None = None,
        secrets_path: Any | None = None,
    ) -> Any:
        """Carga configuración pública y une credenciales desde secretos locales o .env."""
        cfg_candidates = [
            Path(config_path) if config_path else None,
            Path(__file__).resolve().parent / "zephyr_config.json",
            Path("iot_machine_learning/infrastructure/adapters/market/zephyr/config/zephyr_config.json"),
            Path("iot_machine_learning/config/zephyr_config.json"),
            Path("config/zephyr_config.json"),
        ]
        data: dict[str, Any] = {}
        for p in cfg_candidates:
            if p and p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                break

        sec_candidates = [
            Path(secrets_path) if secrets_path else None,
            Path(__file__).resolve().parent / "zephyr_secrets.json",
            Path("iot_machine_learning/infrastructure/adapters/market/zephyr/config/zephyr_secrets.json"),
            Path("iot_machine_learning/config/zephyr_secrets.json"),
            Path("config/zephyr_secrets.json"),
            Path(".secrets.json"),
        ]
        sec_data: dict[str, Any] = {}
        for sp in sec_candidates:
            if sp and sp.exists():
                sec_data = json.loads(sp.read_text(encoding="utf-8"))
                break

        for k, v in sec_data.items():
            if v:
                data[k] = v

        env_map = {
            "ALPACA_API_KEY": "alpaca_api_key",
            "ALPACA_SECRET_KEY": "alpaca_secret_key",
            "BINANCE_API_KEY": "binance_api_key",
            "BINANCE_API_SECRET": "binance_api_secret",
            "WEAVIATE_URL": "weaviate_url",
            "WEAVIATE_API_KEY": "weaviate_api_key",
        }
        for env_var, field_name in env_map.items():
            val = os.getenv(env_var)
            if val and not data.get(field_name):
                data[field_name] = val

        return cls(**data)
