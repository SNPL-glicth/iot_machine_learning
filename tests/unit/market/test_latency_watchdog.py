"""Unit tests for the latency watchdog and pipeline instrumentation."""

from __future__ import annotations

import logging
from collections import deque
import numpy as np
import pytest

from infrastructure.adapters.market.zephyr.config.bot_config import LiveBotConfig
from infrastructure.adapters.market.zephyr.telemetry.builder import perform_health_check


@pytest.mark.asyncio
async def test_health_check_latency_calculation_and_watchdog_warning(caplog):
    """Verifies that perform_health_check uses real samples and alerts when p99 > max_latency_ms."""
    config = LiveBotConfig(symbol="BTC/USD", max_latency_ms=25.0)
    samples: deque[float] = deque(maxlen=1000)

    # 1. Zero/Empty samples fallback
    empty_health = await perform_health_check(
        state=type("State", (), {"last_phi_moe": 0.5, "last_lambda_t": 0.1, "last_phi_ritmo": 0.8, "current_position": 0.0, "trades_count": 0, "last_error": None})(),
        config=config,
        feed=None,
        latency_samples=samples,
        running=True,
    )
    assert empty_health["latency_p50_ms"] == 0.0
    assert empty_health["latency_p99_ms"] == 0.0

    # 2. Add realistic samples below budget (P50 ~ 11.3ms, P99 ~ 22.6ms)
    realistic_data = [11.0, 11.3, 11.5, 12.0, 15.0, 22.0, 22.6]
    samples.extend(realistic_data)

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        healthy_check = await perform_health_check(
            state=type("State", (), {"last_phi_moe": 0.5, "last_lambda_t": 0.1, "last_phi_ritmo": 0.8, "current_position": 0.0, "trades_count": 0, "last_error": None})(),
            config=config,
            feed=None,
            latency_samples=samples,
            running=True,
        )
        assert healthy_check["latency_p50_ms"] == pytest.approx(12.0, abs=1.0)
        assert healthy_check["latency_p99_ms"] <= config.max_latency_ms
        assert "High latency detected" not in caplog.text

    # 3. Add degraded samples that exceed max_latency_ms (e.g. 35ms > 25.0ms)
    samples.extend([30.0, 35.0, 45.0, 50.0])
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        degraded_check = await perform_health_check(
            state=type("State", (), {"last_phi_moe": 0.5, "last_lambda_t": 0.1, "last_phi_ritmo": 0.8, "current_position": 0.0, "trades_count": 0, "last_error": None})(),
            config=config,
            feed=None,
            latency_samples=samples,
            running=True,
        )
        assert degraded_check["latency_p99_ms"] > config.max_latency_ms
        assert "High latency detected" in caplog.text
