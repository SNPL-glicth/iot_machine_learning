"""ConfigManager: Segregated configuration and credentials orchestrator for Zephyr."""
from __future__ import annotations

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from iot_machine_learning.infrastructure.adapters.market.zephyr.config.account_manager import (
    AccountCredentials, AccountManager, AccountProfile,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.config.bot_config import LiveBotConfig

logger = logging.getLogger(__name__)

SECRET_KEYS = (
    "alpaca_api_key", "alpaca_secret_key", "binance_api_key",
    "binance_api_secret", "weaviate_api_key",
)


class ConfigManager:
    """Orchestrates segregated loading of public configuration, secrets, and profiles."""
    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        secrets_path: Optional[Path | str] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        self.config_path = Path(config_path) if config_path else base_dir / "zephyr_config.json"
        self.secrets_path = Path(secrets_path) if secrets_path else base_dir / "zephyr_secrets.json"
        self._account_manager = AccountManager()
        self._raw_config: Dict[str, Any] = {}
        self._raw_secrets: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        self._raw_config = self._load_public_json(self.config_path)
        self._raw_secrets = self._load_secrets_json(self.secrets_path)
        self._sync_profiles()

    def _load_public_json(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return {k: v for k, v in data.items() if k not in SECRET_KEYS}
            except Exception as e:
                logger.error("Failed loading public config %s: %s", path, e)
        return {}

    def _load_secrets_json(self, path: Path) -> Dict[str, Any]:
        secrets: Dict[str, Any] = {}
        if path.exists():
            try:
                secrets = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Could not read secrets file %s: %s", path, e)
        # Fallback to environment variables
        env_map = {
            "ALPACA_API_KEY": "alpaca_api_key", "ALPACA_SECRET_KEY": "alpaca_secret_key",
            "BINANCE_API_KEY": "binance_api_key", "BINANCE_API_SECRET": "binance_api_secret",
            "WEAVIATE_API_KEY": "weaviate_api_key", "WEAVIATE_URL": "weaviate_url",
        }
        for env_k, target_k in env_map.items():
            val = os.getenv(env_k)
            if val and not secrets.get(target_k):
                secrets[target_k] = val
        return secrets

    def _sync_profiles(self) -> None:
        profiles_raw = self._raw_config.get("profiles", [])
        prof_secrets = self._raw_secrets.get("profiles", {})
        for p_data in profiles_raw:
            p_name = p_data.get("profile_name", "default")
            prof = AccountProfile(
                profile_name=p_name,
                broker=p_data.get("broker", "alpaca"),
                symbol=p_data.get("symbol", "SPY"),
                symbols=p_data.get("symbols", [p_data.get("symbol", "SPY")]),
                testnet=p_data.get("testnet", True),
                api_base_url=p_data.get("api_base_url"),
                data_feed=p_data.get("data_feed", "iex"),
                max_position_pct=p_data.get("max_position_pct", 0.05),
                max_trade_loss_usd=p_data.get("max_trade_loss_usd", 10.0),
                lot_size=p_data.get("lot_size", 0.00001),
                extra_params=p_data.get("extra_params", {}),
            )
            creds = None
            if p_name in prof_secrets:
                s = prof_secrets[p_name]
                creds = AccountCredentials(
                    api_key=s.get("api_key") or s.get("alpaca_api_key") or s.get("binance_api_key", ""),
                    secret_key=s.get("secret_key") or s.get("alpaca_secret_key") or s.get("binance_api_secret", ""),
                    api_base_url=s.get("api_base_url"),
                )
            self._account_manager.register_profile(prof, creds)

    @property
    def account_manager(self) -> AccountManager:
        return self._account_manager

    def load_config(self, profile_name: Optional[str] = None) -> LiveBotConfig:
        merged = dict(self._raw_config)
        merged.pop("profiles", None)
        for k in SECRET_KEYS:
            if k in self._raw_secrets and self._raw_secrets[k]:
                merged[k] = self._raw_secrets[k]
        base_cfg = LiveBotConfig(**merged)
        if profile_name:
            return self._account_manager.create_bot_config(profile_name, base_cfg)
        return base_cfg

    def save_public_config(self, config: LiveBotConfig, target_path: Optional[Path] = None) -> None:
        p = target_path or self.config_path
        p.parent.mkdir(parents=True, exist_ok=True)
        pub_dict = config.to_public_dict()
        if "profiles" in self._raw_config:
            pub_dict["profiles"] = self._raw_config["profiles"]
        p.write_text(json.dumps(pub_dict, indent=2), encoding="utf-8")
        logger.info("Saved sanitized public config to %s", p)

    def save_secrets(self, secrets: Dict[str, Any], target_path: Optional[Path] = None) -> None:
        p = target_path or self.secrets_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(secrets, indent=2), encoding="utf-8")
        try:
            os.chmod(p, 0o600)
        except Exception:
            pass
        self._raw_secrets = dict(secrets)

    @staticmethod
    def obfuscate_key(val: Optional[str]) -> str:
        if not val:
            return ""
        s = val
        return f"{s[:4]}******{s[-4:]}" if len(s) >= 8 else "******"
