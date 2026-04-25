"""Tests for IMP-5 Outcome hardening.

Covers:
* extra cannot be mutated post-construction (MappingProxyType).
* extra source dict mutation does not leak into the Outcome.
* Constructor invariants: kind, confidence, trend, anomaly_score.
* with_extra(**kw) returns a new instance, preserves originals.
* with_updates(**fields) re-runs invariants.
* to_dict schema unchanged.
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.entities.explainability.explanation import (
    Outcome,
)


class TestExtraImmutability:
    def test_extra_cannot_be_mutated_after_construction(self) -> None:
        o = Outcome(extra={"a": 1})
        with pytest.raises(TypeError):
            o.extra["a"] = 2  # type: ignore[index]
        with pytest.raises(TypeError):
            o.extra["b"] = 3  # type: ignore[index]

    def test_extra_source_dict_mutation_does_not_leak(self) -> None:
        src = {"a": 1}
        o = Outcome(extra=src)
        src["a"] = 999
        src["new"] = "leak"
        assert dict(o.extra) == {"a": 1}

    def test_extra_empty_by_default(self) -> None:
        o = Outcome()
        assert dict(o.extra) == {}


class TestConstructorInvariants:
    def test_invalid_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            Outcome(kind="bogus")

    def test_allowed_kinds_accepted(self) -> None:
        for k in ("prediction", "anomaly", "prediction+anomaly", "text_analysis", "analysis"):
            Outcome(kind=k)  # must not raise

    def test_confidence_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Outcome(confidence=-0.01)

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Outcome(confidence=1.01)

    def test_confidence_boundaries_accepted(self) -> None:
        Outcome(confidence=0.0)
        Outcome(confidence=1.0)

    def test_invalid_trend_raises(self) -> None:
        with pytest.raises(ValueError, match="trend"):
            Outcome(trend="sideways")

    def test_anomaly_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="anomaly_score"):
            Outcome(anomaly_score=1.5)
        with pytest.raises(ValueError, match="anomaly_score"):
            Outcome(anomaly_score=-0.1)

    def test_anomaly_score_none_accepted(self) -> None:
        Outcome(anomaly_score=None)

    def test_is_anomaly_without_score_permitted(self) -> None:
        # Does not raise; emits DEBUG log (not asserted here).
        Outcome(kind="anomaly", is_anomaly=True, anomaly_score=None)


class TestWithExtra:
    def test_returns_new_instance_with_merge(self) -> None:
        a = Outcome(extra={"x": 1, "y": 2})
        b = a.with_extra(y=99, z=3)
        assert a is not b
        assert dict(b.extra) == {"x": 1, "y": 99, "z": 3}

    def test_does_not_mutate_original(self) -> None:
        a = Outcome(extra={"x": 1})
        _ = a.with_extra(x=2, y=3)
        assert dict(a.extra) == {"x": 1}

    def test_returns_outcome_with_frozen_extra(self) -> None:
        a = Outcome()
        b = a.with_extra(foo="bar")
        with pytest.raises(TypeError):
            b.extra["foo"] = "mutated"  # type: ignore[index]


class TestWithUpdates:
    def test_changes_only_specified_fields(self) -> None:
        a = Outcome(kind="prediction", confidence=0.5, trend="up")
        b = a.with_updates(confidence=0.9)
        assert b.confidence == 0.9
        assert b.kind == "prediction"
        assert b.trend == "up"

    def test_reruns_invariants(self) -> None:
        a = Outcome(kind="prediction")
        with pytest.raises(ValueError, match="kind"):
            a.with_updates(kind="unknown_kind")


class TestSerializationBackCompat:
    def test_to_dict_unchanged_schema(self) -> None:
        o = Outcome(
            kind="prediction",
            predicted_value=42.0,
            confidence=0.87,
            trend="up",
            is_anomaly=False,
        )
        d = o.to_dict()
        assert d == {
            "kind": "prediction",
            "confidence": 0.87,
            "trend": "up",
            "is_anomaly": False,
            "predicted_value": 42.0,
        }

    def test_to_dict_with_extra(self) -> None:
        o = Outcome(extra={"k": "v"})
        d = o.to_dict()
        assert d["extra"] == {"k": "v"}

    def test_to_dict_with_anomaly(self) -> None:
        o = Outcome(
            kind="anomaly", confidence=0.3, is_anomaly=True, anomaly_score=0.91
        )
        d = o.to_dict()
        assert d["is_anomaly"] is True
        assert d["anomaly_score"] == 0.91
