"""AccountManager: Multi-account and multi-broker profile architecture for Zephyr."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Literal, Optional

from iot_machine_learning.infrastructure.adapters.market.zephyr.config.bot_config import LiveBotConfig


@dataclass
class AccountCredentials:
    """Segregated credentials container for a broker account."""
    api_key: str = ""
    secret_key: str = ""
    passphrase: Optional[str] = None
    api_base_url: Optional[str] = None

    def masked_key(self) -> str:
        if not self.api_key:
            return ""
        return f"{self.api_key[:4]}******{self.api_key[-4:]}" if len(self.api_key) >= 8 else "******"

    def masked_secret(self) -> str:
        if not self.secret_key:
            return ""
        return f"{self.secret_key[:4]}******{self.secret_key[-4:]}" if len(self.secret_key) >= 8 else "******"


@dataclass
class AccountProfile:
    """Defines an execution account profile across brokers."""
    profile_name: str
    broker: Literal["alpaca", "binance"] = "alpaca"
    symbol: str = "SPY"
    symbols: List[str] = field(default_factory=list)
    testnet: bool = True
    api_base_url: Optional[str] = None
    data_feed: str = "iex"
    max_position_pct: float = 0.05
    max_trade_loss_usd: float = 10.0
    lot_size: float = 0.00001
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbols:
            self.symbols = [self.symbol.upper()]
        else:
            self.symbols = [s.upper() for s in self.symbols]


class AccountManager:
    """Manages multi-account profiles and secure credential binding."""
    def __init__(
        self,
        profiles: Optional[List[AccountProfile]] = None,
        credentials_map: Optional[Dict[str, AccountCredentials]] = None,
    ) -> None:
        self._profiles: Dict[str, AccountProfile] = {}
        self._credentials: Dict[str, AccountCredentials] = dict(credentials_map or {})
        if profiles:
            for p in profiles:
                self.register_profile(p)

    def register_profile(
        self,
        profile: AccountProfile,
        credentials: Optional[AccountCredentials] = None,
    ) -> None:
        self._profiles[profile.profile_name] = profile
        if credentials:
            self._credentials[profile.profile_name] = credentials

    def list_profiles(self) -> List[str]:
        return list(self._profiles.keys())

    def get_profile(self, profile_name: str) -> AccountProfile:
        if profile_name not in self._profiles:
            raise KeyError(f"Account profile '{profile_name}' not found. Available: {self.list_profiles()}")
        return self._profiles[profile_name]

    def get_credentials(self, profile_name: str) -> Optional[AccountCredentials]:
        if profile_name in self._credentials:
            return self._credentials[profile_name]
        profile = self._profiles.get(profile_name)
        if not profile:
            return None
        # Fallback to environment variables
        if profile.broker == "alpaca":
            k = os.getenv("ALPACA_API_KEY", "")
            s = os.getenv("ALPACA_SECRET_KEY", "")
            u = os.getenv("ALPACA_API_BASE_URL", profile.api_base_url or "https://paper-api.alpaca.markets/v2")
            if k and s:
                return AccountCredentials(api_key=k, secret_key=s, api_base_url=u)
        elif profile.broker == "binance":
            k = os.getenv("BINANCE_API_KEY", "")
            s = os.getenv("BINANCE_API_SECRET", "")
            if k and s:
                return AccountCredentials(api_key=k, secret_key=s)
        return None

    def set_credentials(self, profile_name: str, credentials: AccountCredentials) -> None:
        self._credentials[profile_name] = credentials

    def create_bot_config(
        self,
        profile_name: str,
        base_config: Optional[LiveBotConfig] = None,
    ) -> LiveBotConfig:
        profile = self.get_profile(profile_name)
        creds = self.get_credentials(profile_name)

        cfg_dict: Dict[str, Any] = {}
        if base_config is not None:
            cfg_dict = {k: v for k, v in base_config.__dict__.items() if not k.startswith("_")}

        cfg_dict["broker"] = profile.broker
        cfg_dict["symbol"] = profile.symbol
        cfg_dict["symbols"] = list(profile.symbols)
        cfg_dict["testnet"] = profile.testnet
        cfg_dict["max_position_pct"] = profile.max_position_pct
        cfg_dict["max_trade_loss_usd"] = profile.max_trade_loss_usd
        cfg_dict["lot_size"] = profile.lot_size

        if profile.broker == "alpaca":
            base_url = (creds.api_base_url if creds and creds.api_base_url else None) or profile.api_base_url
            if not base_url:
                base_url = "https://paper-api.alpaca.markets/v2" if profile.testnet else "https://api.alpaca.markets/v2"
            cfg_dict["alpaca_api_base_url"] = base_url
            cfg_dict["alpaca_data_feed"] = profile.data_feed
            if creds:
                cfg_dict["alpaca_api_key"] = creds.api_key
                cfg_dict["alpaca_secret_key"] = creds.secret_key
        elif profile.broker == "binance":
            if creds:
                cfg_dict["binance_api_key"] = creds.api_key
                cfg_dict["binance_api_secret"] = creds.secret_key

        for k, v in profile.extra_params.items():
            cfg_dict[k] = v

        return LiveBotConfig(**cfg_dict)
