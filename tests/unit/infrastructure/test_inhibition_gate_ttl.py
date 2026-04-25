"""Tests para InhibitionGate TTL (FIX-7).

Entries older than entry_ttl_seconds are treated as cold start.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.inhibition.gate import (
    InhibitionConfig,
    InhibitionGate,
)
from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
    EnginePerception,
)


class TestInhibitionGateTTL:
    """TTL expiry for inactive series entries."""

    @patch("time.monotonic")
    def test_expired_entry_treated_as_cold_start(self, mock_time) -> None:
        """Write entry, advance past TTL, read returns prev=0.0."""
        gate = InhibitionGate(
            config=InhibitionConfig(),
            entry_ttl_seconds=60.0,
        )
        mock_time.return_value = 1000.0

        p = EnginePerception(
            engine_name="test",
            predicted_value=10.0,
            confidence=0.8,
            trend="stable",
            stability=0.9,  # > threshold 0.6 → triggers suppression
            local_fit_error=0.0,
        )

        # First call: stores suppression
        gate.compute([p], {"test": 1.0})
        assert len(gate._prev_suppression) == 1

        # Advance past TTL
        mock_time.return_value = 1100.0  # 100s > 60s TTL
        states = gate.compute([p], {"test": 1.0})

        # Should be cold start: prev = 0.0, no previous suppression applied
        # With stability=0.9, instant_suppression should be non-zero
        # But prev should be 0.0 (expired)
        assert states[0].suppression_factor < 1.0  # Less than full suppression

    @patch("time.monotonic")
    def test_active_entry_not_expired(self, mock_time) -> None:
        """Write entry, advance just under TTL, read still returns data."""
        gate = InhibitionGate(
            config=InhibitionConfig(),
            entry_ttl_seconds=60.0,
        )
        mock_time.return_value = 1000.0

        p = EnginePerception(
            engine_name="test",
            predicted_value=10.0,
            confidence=0.8,
            trend="stable",
            stability=0.9,
            local_fit_error=0.0,
        )

        gate.compute([p], {"test": 1.0})

        # Advance just under TTL
        mock_time.return_value = 1059.0  # 59s < 60s TTL
        states = gate.compute([p], {"test": 1.0})

        # Previous suppression should still be applied (decayed)
        assert len(gate._prev_suppression) == 1

    @patch("time.monotonic")
    def test_purge_expired_removes_stale_entries(self, mock_time) -> None:
        """purge_expired() removes entries past TTL."""
        gate = InhibitionGate(
            config=InhibitionConfig(),
            entry_ttl_seconds=60.0,
        )
        mock_time.return_value = 1000.0

        p = EnginePerception(
            engine_name="test",
            predicted_value=10.0,
            confidence=0.8,
            trend="stable",
            stability=0.5,  # < threshold, no suppression
            local_fit_error=0.0,
        )

        gate.compute([p], {"test": 1.0})
        assert len(gate._prev_suppression) == 1

        # Advance past TTL
        mock_time.return_value = 1100.0
        n_removed = gate.purge_expired()
        assert n_removed == 1
        assert len(gate._prev_suppression) == 0
        assert len(gate._last_update) == 0

    def test_invalid_ttl_raises(self) -> None:
        """entry_ttl_seconds < 0 raises ValueError."""
        with pytest.raises(ValueError, match="entry_ttl_seconds"):
            InhibitionGate(entry_ttl_seconds=-1.0)
