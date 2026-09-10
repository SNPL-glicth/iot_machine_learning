"""Test Alpaca configuration in LiveBotConfig."""

from __future__ import annotations

import os
from iot_machine_learning.infrastructure.adapters.market.live_config import LiveBotConfig


def test_default_binance_config():
    """Test default Binance configuration."""
    cfg = LiveBotConfig()
    assert cfg.broker == "binance"
    assert cfg.testnet is True
    assert cfg.is_paper_trading is True
    assert cfg.is_live_trading is False


def test_alpaca_paper_config_from_env():
    """Test Alpaca Paper configuration loaded from environment."""
    cfg = LiveBotConfig(
        broker="alpaca",
        alpaca_api_key=os.getenv("ALPACA_API_KEY"),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY"),
        alpaca_api_base_url=os.getenv("ALPACA_API_BASE_URL"),
        alpaca_data_feed=os.getenv("ALPACA_DATA_FEED", "iex"),
    )
    assert cfg.broker == "alpaca"
    assert cfg.alpaca_api_base_url == "https://paper-api.alpaca.markets"
    assert cfg.alpaca_data_feed == "iex"
    assert cfg.is_paper_trading is True
    assert cfg.is_live_trading is False


def test_alpaca_rejects_live_url():
    """Test that Alpaca config rejects live trading URL."""
    try:
        LiveBotConfig(
            broker="alpaca",
            alpaca_api_key="test",
            alpaca_secret_key="test",
            alpaca_api_base_url="https://api.alpaca.markets/v2",
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "paper-api.alpaca.markets" in str(e)


def test_alpaca_requires_credentials():
    """Test that Alpaca config requires API credentials."""
    try:
        LiveBotConfig(broker="alpaca")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "alpaca_api_key" in str(e) or "alpaca_secret_key" in str(e)


def test_alpaca_rejects_invalid_data_feed():
    """Test that Alpaca config rejects invalid data feed."""
    try:
        LiveBotConfig(
            broker="alpaca",
            alpaca_api_key="test",
            alpaca_secret_key="test",
            alpaca_api_base_url="https://paper-api.alpaca.markets/v2",
            alpaca_data_feed="invalid",
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "data_feed" in str(e).lower() or "iex" in str(e).lower() or "sip" in str(e).lower()


def test_binance_testnet_is_paper():
    """Test that Binance testnet is detected as paper trading."""
    cfg = LiveBotConfig(broker="binance", testnet=True)
    assert cfg.is_paper_trading is True
    assert cfg.is_live_trading is False


def test_binance_mainnet_is_live():
    """Test that Binance mainnet is detected as live trading."""
    cfg = LiveBotConfig(broker="binance", testnet=False)
    assert cfg.is_paper_trading is False
    assert cfg.is_live_trading is True
