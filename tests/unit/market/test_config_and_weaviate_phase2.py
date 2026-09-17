"""Unit tests for Phase 2: Configuration Segregation and Weaviate Cognitive Memory.

Validates:
1. Credential segregation: secrets never leak to public config or dashboard
2. ConfigSerializationMixin safe dictionaries and load_with_secrets
3. Commands extraction sanitization
4. WeaviateTelemetryStore query resilience
5. Strict <= 180 lines limit per file
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from iot_machine_learning.infrastructure.adapters.market.zephyr.config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.zephyr.telemetry.commands import (
    apply_bot_config_update,
    extract_bot_config_dict,
)
from iot_machine_learning.infrastructure.adapters.persistence.weaviate_telemetry import (
    WeaviateTelemetryStore,
)


def test_credential_segregation_in_safe_dicts():
    """Verify secrets are redacted in safe dict and omitted in public dict."""
    config = LiveBotConfig(
        symbol="SPY",
        alpaca_api_key="TEST_KEY_SECRET",
        alpaca_secret_key="TEST_SECRET_VALUE",
    )

    safe_dict = config.to_safe_dict()
    assert safe_dict["alpaca_api_key"] == "***REDACTED***"
    assert safe_dict["alpaca_secret_key"] == "***REDACTED***"

    pub_dict = config.to_public_dict()
    assert "alpaca_api_key" not in pub_dict
    assert "alpaca_secret_key" not in pub_dict
    assert pub_dict["symbol"] == "SPY"


def test_extract_bot_config_dict_sanitization():
    """Verify extract_bot_config_dict never leaks plain API keys."""
    config = LiveBotConfig(
        symbol="SPY",
        alpaca_api_key="SUPER_SECRET_ALPACA_KEY",
        alpaca_secret_key="SUPER_SECRET_ALPACA_SECRET",
    )

    extracted = extract_bot_config_dict(config)
    assert extracted["alpaca_api_key"] == "***REDACTED***"
    assert extracted["alpaca_secret_key"] == "***REDACTED***"


def test_apply_bot_config_update_segregation(tmp_path):
    """Verify live updates do not return raw secrets and update public config safely."""
    runner = MagicMock()
    runner.config = LiveBotConfig(symbol="SPY", alpaca_api_key="OLD_KEY", alpaca_secret_key="OLD_SEC")

    updates = {
        "phi_moe_threshold": 0.52,
        "alpaca_api_key": "NEW_SENSITIVE_KEY",
    }

    result = apply_bot_config_update(runner, updates)
    assert runner.config.phi_moe_threshold == 0.52
    assert result["alpaca_api_key"] == "***REDACTED***"


def test_load_with_secrets_merges_properly():
    """Verify load_with_secrets loads public params and secrets without collision."""
    loaded = LiveBotConfig.load_with_secrets()
    assert isinstance(loaded, LiveBotConfig)
    assert loaded.symbol in ("SPY", "BTCUSDT")
    assert loaded.master_shadow_mode is False
    assert loaded.dry_run is False


@pytest.mark.asyncio
async def test_weaviate_telemetry_queries_offline_resilience():
    """Verify Weaviate query methods return empty lists gracefully on network failure."""
    store = WeaviateTelemetryStore(url="http://127.0.0.1:9999")  # Non-existent port

    telemetry = await store.query_recent_telemetry(symbol="SPY", limit=10)
    assert isinstance(telemetry, list)
    assert len(telemetry) == 0

    executions = await store.query_recent_executions(symbol="SPY", limit=10)
    assert isinstance(executions, list)
    assert len(executions) == 0

    await store.flush_and_close()


def test_phase2_file_line_limits():
    """Verify all Phase 2 files strictly respect the <= 180 lines limit."""
    base_dir = Path(__file__).resolve().parents[3] / "infrastructure" / "adapters"
    files = [
        base_dir / "market" / "zephyr" / "config" / "bot_config.py",
        base_dir / "market" / "zephyr" / "config" / "validator.py",
        base_dir / "market" / "zephyr" / "telemetry" / "commands.py",
        base_dir / "market" / "zephyr" / "telemetry" / "server.py",
        base_dir / "persistence" / "weaviate_telemetry.py",
    ]
    for f in files:
        assert f.exists(), f"File {f} not found"
        lines = len(f.read_text(encoding="utf-8").splitlines())
        assert lines <= 180, f"{f.name} exceeds 180 lines: {lines}"
