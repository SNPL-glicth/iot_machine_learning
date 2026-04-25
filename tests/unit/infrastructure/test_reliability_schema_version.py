"""Tests para EngineReliabilityTracker schema version (FIX-8).

Schema mismatch → discard stored posterior, return uninformative prior.
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.reliability.engine_reliability_tracker import (
    EngineReliabilityTracker,
)
from iot_machine_learning.infrastructure.ml.cognitive.error_store import (
    EngineErrorStore,
)


class _FakeErrorStore:
    """Minimal EngineErrorStore stub."""

    def get_percentile(self, series_id: str, engine_name: str, percentile: float) -> float:
        return 1.0  # Always enough history

    def record(self, series_id: str, engine_name: str, error: float) -> None:
        pass


class TestSchemaVersion:
    """Schema version compatibility for stored posteriors."""

    def test_load_wrong_schema_returns_prior(self) -> None:
        """Stored hash with schema_version='0' → return alpha=1.0, beta=1.0."""
        store = _FakeErrorStore()
        tracker = EngineReliabilityTracker(error_store=store)

        # Inject old-schema data directly into memory
        tracker._memory[("s1", "e1")] = (5.0, 2.0, "0")

        alpha, beta = tracker._load("s1", "e1")
        assert alpha == 1.0
        assert beta == 1.0

    def test_load_correct_schema_returns_stored(self) -> None:
        """Stored hash with matching schema_version → return stored values."""
        store = _FakeErrorStore()
        tracker = EngineReliabilityTracker(error_store=store)

        tracker._memory[("s1", "e1")] = (5.0, 2.0, tracker._schema_version)

        alpha, beta = tracker._load("s1", "e1")
        assert alpha == 5.0
        assert beta == 2.0

    def test_save_writes_schema_version(self) -> None:
        """_save stores (alpha, beta, schema_version) in memory."""
        store = _FakeErrorStore()
        tracker = EngineReliabilityTracker(error_store=store)

        tracker._save("s1", "e1", 3.0, 7.0)

        entry = tracker._memory[("s1", "e1")]
        assert len(entry) == 3
        assert entry[2] == tracker._schema_version

    def test_roundtrip_via_record_outcome(self) -> None:
        """record_outcome → _save; _load returns same values."""
        store = _FakeErrorStore()
        tracker = EngineReliabilityTracker(error_store=store)

        tracker.record_outcome("s1", "e1", 2.0)  # error > threshold(1.0) → beta += 1
        alpha, beta = tracker._load("s1", "e1")
        assert alpha == 1.0
        assert beta == 2.0

        tracker.record_outcome("s1", "e1", 0.5)  # error < threshold(1.0) → alpha += 1
        alpha, beta = tracker._load("s1", "e1")
        assert alpha == 2.0
        assert beta == 2.0
