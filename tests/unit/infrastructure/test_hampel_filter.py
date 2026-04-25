"""Tests for hampel_filter (IMP-2).

Covers:
* No-op when < min_perceptions.
* No-op when MAD = 0 (all identical).
* Positive and negative outliers rejected.
* Clustered-but-noisy values kept.
* Order preservation.
* Custom k affects strictness.
* Rejected tuples contain z-scores.
* k <= 0 disables filtering.
"""

from __future__ import annotations

from typing import List

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
    EnginePerception,
)
from iot_machine_learning.infrastructure.ml.cognitive.fusion import (
    HampelResult,
    hampel_filter,
)


def _perc(name: str, value: float, conf: float = 0.8) -> EnginePerception:
    return EnginePerception(
        engine_name=name, predicted_value=value, confidence=conf, trend="stable"
    )


class TestHampelBasics:
    def test_empty_input(self) -> None:
        r = hampel_filter([])
        assert r.kept == []
        assert r.rejected == []
        assert r.mad == 0.0

    def test_no_filter_below_min_count(self) -> None:
        perceptions = [_perc("a", 10.0), _perc("b", 1000.0)]  # only 2
        r = hampel_filter(perceptions, min_perceptions=3)
        assert [p.engine_name for p in r.kept] == ["a", "b"]
        assert r.rejected == []

    def test_no_filter_when_mad_zero(self) -> None:
        # All identical → MAD=0, nothing to reject.
        perceptions = [_perc("a", 5.0), _perc("b", 5.0), _perc("c", 5.0)]
        r = hampel_filter(perceptions)
        assert len(r.kept) == 3
        assert r.rejected == []
        assert r.mad == 0.0
        assert r.median == 5.0


class TestHampelOutlierRejection:
    def test_positive_outlier_rejected(self) -> None:
        # [10, 11, 100] → median=11, deviations=[1, 0, 89], MAD=1
        perceptions = [_perc("a", 10.0), _perc("b", 11.0), _perc("c", 100.0)]
        r = hampel_filter(perceptions, k=3.0)
        kept_names = [p.engine_name for p in r.kept]
        assert "c" not in kept_names
        assert kept_names == ["a", "b"]  # order preserved
        assert len(r.rejected) == 1
        assert r.rejected[0][0] == "c"
        assert r.rejected[0][1] == 100.0

    def test_negative_outlier_rejected(self) -> None:
        perceptions = [_perc("a", 10.0), _perc("b", 11.0), _perc("c", -50.0)]
        r = hampel_filter(perceptions, k=3.0)
        assert [p.engine_name for p in r.kept] == ["a", "b"]
        assert r.rejected[0][0] == "c"

    def test_all_within_threshold_kept(self) -> None:
        # Noisy but clustered values, all within 3·1.4826·MAD of median.
        perceptions = [
            _perc("a", 10.0),
            _perc("b", 10.5),
            _perc("c", 9.5),
            _perc("d", 10.2),
            _perc("e", 9.8),
        ]
        r = hampel_filter(perceptions, k=3.0)
        assert len(r.kept) == 5
        assert r.rejected == []


class TestHampelSemantics:
    def test_order_preserved(self) -> None:
        perceptions = [
            _perc("x", 10.0),
            _perc("y", 500.0),  # outlier
            _perc("z", 11.0),
            _perc("w", 10.5),
        ]
        r = hampel_filter(perceptions, k=3.0)
        kept_order = [p.engine_name for p in r.kept]
        assert kept_order == ["x", "z", "w"]

    def test_custom_k_tighter(self) -> None:
        perceptions = [_perc("a", 10.0), _perc("b", 11.0), _perc("c", 14.0)]
        # Median=11, MAD=1, σ̂≈1.4826. k=1 → threshold=1.4826.
        # |14-11|=3 > 1.4826 → rejected; |10-11|=1 < 1.4826 → kept.
        r = hampel_filter(perceptions, k=1.0)
        assert "c" not in [p.engine_name for p in r.kept]

    def test_custom_k_looser(self) -> None:
        perceptions = [_perc("a", 10.0), _perc("b", 11.0), _perc("c", 100.0)]
        # With very loose k, nothing rejected even for 100.
        r = hampel_filter(perceptions, k=100.0)
        assert len(r.kept) == 3
        assert r.rejected == []

    def test_k_nonpositive_disables_filter(self) -> None:
        perceptions = [_perc("a", 10.0), _perc("b", 11.0), _perc("c", 10_000.0)]
        r = hampel_filter(perceptions, k=0.0)
        assert len(r.kept) == 3
        r2 = hampel_filter(perceptions, k=-1.0)
        assert len(r2.kept) == 3

    def test_rejected_contains_z_score(self) -> None:
        perceptions = [_perc("a", 10.0), _perc("b", 11.0), _perc("c", 100.0)]
        r = hampel_filter(perceptions, k=3.0)
        assert len(r.rejected) == 1
        name, value, z = r.rejected[0]
        assert name == "c"
        assert value == 100.0
        assert z > 3.0  # outside the k=3 threshold by construction

    def test_to_dict_schema(self) -> None:
        perceptions = [_perc("a", 10.0), _perc("b", 11.0), _perc("c", 100.0)]
        r = hampel_filter(perceptions, k=3.0)
        d = r.to_dict()
        assert set(d.keys()) == {"median", "mad", "rejected"}
        assert d["rejected"][0]["engine"] == "c"
        assert d["rejected"][0]["predicted_value"] == 100.0
        assert "z_score" in d["rejected"][0]
