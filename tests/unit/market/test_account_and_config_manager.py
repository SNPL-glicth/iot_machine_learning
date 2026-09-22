"""Unit tests for Phase 3: ConfigManager and AccountManager Architecture."""
from pathlib import Path
import json
import os
import pytest

from iot_machine_learning.infrastructure.adapters.market.zephyr.config.account_manager import (
    AccountCredentials, AccountManager, AccountProfile,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.config.config_manager import (
    ConfigManager,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.config.loader import (
    load_zephyr_config,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.telemetry.commands import (
    extract_bot_config_dict, obfuscate_secret,
)


def test_account_credentials_masking():
    creds = AccountCredentials(
        api_key="PKH2R77OFHF27PW22TCK2PBBLE",
        secret_key="Famtx4igmRSFjg4tWeMLGtLjRZWGzJkJUyXrqagzVa4S",
    )
    assert creds.masked_key() == "PKH2******BBLE"
    assert creds.masked_secret() == "Famt******Va4S"

    short_creds = AccountCredentials(api_key="TINY", secret_key="123")
    assert short_creds.masked_key() == "******"
    assert short_creds.masked_secret() == "******"


def test_account_profile_and_manager_registration():
    prof = AccountProfile(
        profile_name="test_alpaca",
        broker="alpaca",
        symbol="SPY",
        symbols=["SPY"],
        testnet=True,
    )
    creds = AccountCredentials(api_key="KEY123456", secret_key="SEC123456")
    mgr = AccountManager([prof])
    mgr.set_credentials("test_alpaca", creds)

    assert "test_alpaca" in mgr.list_profiles()
    retrieved_prof = mgr.get_profile("test_alpaca")
    assert retrieved_prof.symbol == "SPY"
    retrieved_creds = mgr.get_credentials("test_alpaca")
    assert retrieved_creds.api_key == "KEY123456"

    bot_cfg = mgr.create_bot_config("test_alpaca")
    assert bot_cfg.broker == "alpaca"
    assert bot_cfg.symbol == "SPY"
    assert bot_cfg.alpaca_api_key == "KEY123456"


def test_account_manager_env_fallback(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "ENV_KEY_12345")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "ENV_SEC_12345")

    prof = AccountProfile(profile_name="env_profile", broker="alpaca", symbol="QQQ")
    mgr = AccountManager([prof])
    creds = mgr.get_credentials("env_profile")
    assert creds is not None
    assert creds.api_key == "ENV_KEY_12345"
    assert creds.secret_key == "ENV_SEC_12345"


def test_config_manager_segregation_and_profile_loading(tmp_path):
    pub_cfg_path = tmp_path / "test_config.json"
    sec_cfg_path = tmp_path / "test_secrets.json"

    pub_data = {
        "broker": "alpaca",
        "symbol": "SPY",
        "symbols": ["SPY"],
        "max_position_pct": 0.05,
        "profiles": [
            {
                "profile_name": "alpaca_paper_01",
                "broker": "alpaca",
                "symbol": "SPY",
                "symbols": ["SPY"],
                "testnet": True,
            },
            {
                "profile_name": "binance_live_01",
                "broker": "binance",
                "symbol": "BTCUSDT",
                "symbols": ["BTCUSDT"],
                "testnet": False,
            },
        ],
    }
    sec_data = {
        "alpaca_api_key": "FALLBACK_KEY_12345",
        "alpaca_secret_key": "FALLBACK_SEC_12345",
        "profiles": {
            "alpaca_paper_01": {
                "api_key": "PROF_ALP_KEY_9999",
                "secret_key": "PROF_ALP_SEC_9999",
            }
        },
    }

    pub_cfg_path.write_text(json.dumps(pub_data), encoding="utf-8")
    sec_cfg_path.write_text(json.dumps(sec_data), encoding="utf-8")

    mgr = ConfigManager(config_path=pub_cfg_path, secrets_path=sec_cfg_path)
    assert set(mgr.account_manager.list_profiles()) == {"alpaca_paper_01", "binance_live_01"}

    # Load specific profile
    alp_cfg = mgr.load_config(profile_name="alpaca_paper_01")
    assert alp_cfg.broker == "alpaca"
    assert alp_cfg.symbol == "SPY"
    assert alp_cfg.alpaca_api_key == "PROF_ALP_KEY_9999"

    # Save public config: verify secrets are NEVER written to public file
    alp_cfg.alpaca_api_key = "DO_NOT_PERSIST_KEY"
    new_pub_path = tmp_path / "saved_public.json"
    mgr.save_public_config(alp_cfg, target_path=new_pub_path)
    saved_text = new_pub_path.read_text(encoding="utf-8")
    assert "DO_NOT_PERSIST_KEY" not in saved_text
    assert "alpaca_api_key" not in json.loads(saved_text)


def test_telemetry_obfuscation_functionality():
    masked = obfuscate_secret("PKH2R77OFHF27PW22TCK2PBBLE")
    assert masked == "PKH2******BBLE"

    from iot_machine_learning.infrastructure.adapters.market.zephyr.config.bot_config import LiveBotConfig
    cfg = LiveBotConfig(symbol="SPY", alpaca_api_key="MY_SECRET_API_KEY", alpaca_secret_key="MY_SECRET_VALUE_12345")
    obfuscated_dict = extract_bot_config_dict(cfg, obfuscate=True)
    assert obfuscated_dict["alpaca_api_key"] == "MY_S******_KEY"
    assert obfuscated_dict["alpaca_secret_key"] == "MY_S******2345"


def test_phase3_files_line_limit():
    base_dir = Path(__file__).resolve().parents[3] / "infrastructure" / "adapters" / "market" / "zephyr"
    files = [
        base_dir / "config" / "account_manager.py",
        base_dir / "config" / "config_manager.py",
        base_dir / "config" / "loader.py",
        base_dir / "config" / "__init__.py",
        base_dir / "telemetry" / "commands.py",
    ]
    for f in files:
        assert f.exists(), f"File {f} not found"
        lines = len(f.read_text(encoding="utf-8").splitlines())
        assert lines <= 180, f"{f.name} exceeds 180 lines: {lines}"
