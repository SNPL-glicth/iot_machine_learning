"""Tests for cognitive/inhibition.py — weight suppression logic."""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.inhibition import (
    InhibitionConfig,
    InhibitionGate,
)
from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
    EnginePerception,
    InhibitionState,
)
from iot_machine_learning.infrastructure.ml.cognitive.error_store import (
    EngineErrorStore,
)
from iot_machine_learning.infrastructure.ml.cognitive.reliability import (
    EngineReliabilityTracker,
)


def _perception(
    name: str = "eng",
    stability: float = 0.0,
    fit_error: float = 0.0,
) -> EnginePerception:
    return EnginePerception(
        engine_name=name,
        predicted_value=10.0,
        confidence=0.8,
        stability=stability,
        local_fit_error=fit_error,
    )


class TestInhibitionGateBasic:

    def test_no_suppression_stable_engine(self) -> None:
        gate = InhibitionGate()
        perceptions = [_perception("a", stability=0.1, fit_error=0.5)]
        weights = {"a": 0.5}
        states = gate.compute(perceptions, weights)
        assert len(states) == 1
        assert states[0].suppression_factor == pytest.approx(0.0)
        assert states[0].inhibited_weight == pytest.approx(0.5)
        assert states[0].inhibition_reason == "none"

    def test_instability_suppression(self) -> None:
        gate = InhibitionGate()
        perceptions = [_perception("a", stability=0.9)]
        weights = {"a": 0.5}
        states = gate.compute(perceptions, weights)
        assert states[0].suppression_factor > 0.0
        assert states[0].inhibited_weight < 0.5
        assert "instability" in states[0].inhibition_reason

    def test_fit_error_suppression(self) -> None:
        gate = InhibitionGate()
        perceptions = [_perception("a", fit_error=20.0)]
        weights = {"a": 0.5}
        states = gate.compute(perceptions, weights)
        assert states[0].suppression_factor > 0.0
        assert "fit_error" in states[0].inhibition_reason

    def test_recent_error_suppression(self) -> None:
        gate = InhibitionGate()
        perceptions = [_perception("a")]
        weights = {"a": 0.5}
        recent = {"a": [50.0, 60.0, 70.0]}
        states = gate.compute(perceptions, weights, recent)
        assert states[0].suppression_factor > 0.0
        assert "recent_error" in states[0].inhibition_reason

    def test_min_weight_floor(self) -> None:
        cfg = InhibitionConfig(min_weight=0.05)
        gate = InhibitionGate(cfg)
        perceptions = [_perception("a", stability=0.99, fit_error=100.0)]
        weights = {"a": 0.5}
        recent = {"a": [100.0] * 10}
        states = gate.compute(perceptions, weights, recent)
        assert states[0].inhibited_weight >= cfg.min_weight

    def test_multiple_engines(self) -> None:
        gate = InhibitionGate()
        perceptions = [
            _perception("stable_eng", stability=0.1),
            _perception("unstable_eng", stability=0.9),
        ]
        weights = {"stable_eng": 0.5, "unstable_eng": 0.5}
        states = gate.compute(perceptions, weights)
        stable = [s for s in states if s.engine_name == "stable_eng"][0]
        unstable = [s for s in states if s.engine_name == "unstable_eng"][0]
        assert stable.inhibited_weight > unstable.inhibited_weight


class TestInhibitionConfig:

    def test_custom_thresholds(self) -> None:
        cfg = InhibitionConfig(
            stability_threshold=0.3,
            fit_error_threshold=1.0,
            recent_error_threshold=2.0,
        )
        gate = InhibitionGate(cfg)
        perceptions = [_perception("a", stability=0.5)]
        weights = {"a": 0.5}
        states = gate.compute(perceptions, weights)
        assert states[0].suppression_factor > 0.0


class TestInhibitionViaReliability:
    """IMP-4b: Beta-Bernoulli path overrides the legacy 3 rules."""

    def _build(self, errors):
        store = EngineErrorStore()
        for e in errors:
            store.record("s1", "a", e)
        tracker = EngineReliabilityTracker(store)
        gate = InhibitionGate(reliability_tracker=tracker)
        return gate, tracker

    def test_reliable_engine_passes_through_at_full_weight(self) -> None:
        gate, tracker = self._build([1.0] * 20)
        for _ in range(20):
            tracker.record_outcome("s1", "a", 0.1)
        # Intentionally extreme perception values \u2014 the 3 legacy rules
        # would suppress. The reliability path must ignore them.
        perceptions = [_perception("a", stability=0.99, fit_error=100.0)]
        states = gate.compute(
            perceptions,
            {"a": 0.5},
            recent_errors={"a": [100.0] * 10},
            series_id="s1",
        )
        assert states[0].inhibited_weight == pytest.approx(0.5)
        assert states[0].inhibition_reason == "none"
        assert states[0].suppression_factor == 0.0

    def test_unreliable_engine_is_hard_excluded(self) -> None:
        gate, tracker = self._build([1.0] * 20)
        for _ in range(200):
            tracker.record_outcome("s1", "a", 99.0)
        # Clean-looking perception \u2014 legacy rules would PASS. The
        # reliability path must exclude because P(broken) > 0.95.
        perceptions = [_perception("a", stability=0.0, fit_error=0.0)]
        states = gate.compute(perceptions, {"a": 0.5}, series_id="s1")
        assert states[0].inhibited_weight == 0.0
        assert states[0].suppression_factor == 1.0
        assert "unreliable" in states[0].inhibition_reason

    def test_fresh_engine_is_reliable_by_default(self) -> None:
        gate, _tracker = self._build([])
        perceptions = [_perception("a")]
        states = gate.compute(perceptions, {"a": 0.5}, series_id="new_series")
        assert states[0].inhibited_weight == pytest.approx(0.5)

    def test_legacy_path_still_works_without_tracker(self) -> None:
        gate = InhibitionGate()  # no tracker
        perceptions = [_perception("a", stability=0.9)]
        states = gate.compute(perceptions, {"a": 0.5})
        assert states[0].suppression_factor > 0.0
