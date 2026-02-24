"""Tests for cognitive/engine_selector.py — weighted fusion logic."""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.engine_selector import (
    WeightedFusion,
)
from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
    EnginePerception,
    InhibitionState,
)


def _perception(
    name: str, value: float, confidence: float = 0.8,
    trend: str = "stable",
) -> EnginePerception:
    return EnginePerception(
        engine_name=name, predicted_value=value,
        confidence=confidence, trend=trend,
    )


def _inh_state(name: str, weight: float) -> InhibitionState:
    return InhibitionState(
        engine_name=name, base_weight=weight,
        inhibited_weight=weight,
    )


class TestWeightedFusionBasic:

    def test_single_engine(self) -> None:
        fusion = WeightedFusion()
        perceptions = [_perception("a", 10.0, 0.9)]
        states = [_inh_state("a", 1.0)]
        val, conf, trend, weights, selected, reason = fusion.fuse(
            perceptions, states)
        assert val == pytest.approx(10.0)
        assert conf == pytest.approx(0.9)
        assert selected == "a"

    def test_equal_weights(self) -> None:
        fusion = WeightedFusion()
        perceptions = [
            _perception("a", 10.0, 0.8),
            _perception("b", 20.0, 0.6),
        ]
        states = [_inh_state("a", 0.5), _inh_state("b", 0.5)]
        val, conf, trend, weights, selected, reason = fusion.fuse(
            perceptions, states)
        assert val == pytest.approx(15.0)
        assert conf == pytest.approx(0.7)

    def test_unequal_weights(self) -> None:
        fusion = WeightedFusion()
        perceptions = [
            _perception("a", 10.0, 0.8),
            _perception("b", 20.0, 0.6),
        ]
        states = [_inh_state("a", 0.75), _inh_state("b", 0.25)]
        val, conf, trend, weights, selected, reason = fusion.fuse(
            perceptions, states)
        # 10*0.75 + 20*0.25 = 12.5
        assert val == pytest.approx(12.5)
        assert selected == "a"

    def test_trend_majority_vote(self) -> None:
        fusion = WeightedFusion()
        perceptions = [
            _perception("a", 10.0, trend="up"),
            _perception("b", 20.0, trend="up"),
            _perception("c", 15.0, trend="down"),
        ]
        states = [
            _inh_state("a", 0.33),
            _inh_state("b", 0.34),
            _inh_state("c", 0.33),
        ]
        _, _, trend, _, _, _ = fusion.fuse(perceptions, states)
        assert trend == "up"

    def test_empty_perceptions(self) -> None:
        fusion = WeightedFusion()
        val, conf, trend, weights, selected, reason = fusion.fuse([], [])
        assert val == 0.0
        assert selected == "none"
        assert reason == "no_engines"

    def test_weights_normalized(self) -> None:
        fusion = WeightedFusion()
        perceptions = [
            _perception("a", 10.0),
            _perception("b", 20.0),
        ]
        states = [_inh_state("a", 3.0), _inh_state("b", 1.0)]
        _, _, _, weights, _, _ = fusion.fuse(perceptions, states)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["a"] == pytest.approx(0.75)
        assert weights["b"] == pytest.approx(0.25)


class TestSelectionReason:

    def test_reason_includes_highest_weight(self) -> None:
        fusion = WeightedFusion()
        perceptions = [_perception("a", 10.0)]
        states = [_inh_state("a", 1.0)]
        _, _, _, _, _, reason = fusion.fuse(perceptions, states)
        assert "highest_weight" in reason

    def test_reason_mentions_inhibited_engines(self) -> None:
        fusion = WeightedFusion()
        perceptions = [
            _perception("a", 10.0),
            _perception("b", 20.0),
        ]
        states = [
            _inh_state("a", 0.8),
            InhibitionState(
                engine_name="b", base_weight=0.5,
                inhibited_weight=0.2, suppression_factor=0.6,
                inhibition_reason="instability=0.9",
            ),
        ]
        _, _, _, _, _, reason = fusion.fuse(perceptions, states)
        assert "inhibited" in reason
