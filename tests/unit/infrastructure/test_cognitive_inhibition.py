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
