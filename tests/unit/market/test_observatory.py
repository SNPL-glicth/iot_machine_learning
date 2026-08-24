"""Tests del Prediction Observatory (FASE 10) — memoria observable pura."""

from __future__ import annotations

import pytest
from iot_machine_learning.domain.entities.market.adaptation.guard import (
    wilson_lower_bound,
)
from iot_machine_learning.domain.entities.market.observatory import (
    ContextLearning,
    ObservationRow,
    calibration_curve,
    dimension_stats,
    evidence_requirement,
    is_degraded,
    learning_curve,
    observatory_summary,
    recency_bands,
    render_observatory,
)
from iot_machine_learning.domain.entities.market.replay.calibration import (
    BucketStatus,
    CalibrationThresholds,
)


def row(
    *,
    rid: int = 0,
    status: str = "rewarded",
    correct: bool | None = True,
    prob: float = 0.6,
    strategy: str = "momentum",
    horizon: int = 3600,
    regime: str | None = "TRENDING",
    emitted: float = 1000.0,
    reward: float | None = 1.0,
    calibration: float | None = 0.2,
    stale: bool = False,
) -> ObservationRow:
    return ObservationRow(
        prediction_id=f"p{rid}",
        emitted_at=emitted,
        strategy=strategy,
        horizon_seconds=horizon,
        regime=regime,
        probability_up=prob,
        direction_correct=correct,
        outcome_return_realized=0.01 if correct else -0.01,
        reward_total=reward,
        calibration_error=calibration,
        status=status,
        data_status="stale" if stale else "replay",
    )


class TestSummary:
    def test_counts_and_accuracy(self) -> None:
        rows = [
            row(rid=0, status="rewarded", correct=True),
            row(rid=1, status="rewarded", correct=True),
            row(rid=2, status="rewarded", correct=False),
            row(rid=3, status="rewarded", correct=False),
            row(rid=4, status="rewarded", correct=False),
            row(rid=5, status="rewarded", correct=False),
            row(rid=6, status="pending"),
            row(rid=7, status="active"),
            row(rid=8, status="invalidated"),
            row(rid=9, status="archived", stale=True),
        ]
        s = observatory_summary(rows)
        assert s.total == 10
        assert s.evaluated == 6
        assert s.pending == 2
        assert s.invalidated == 1
        assert s.archived == 1
        assert s.stale == 1
        assert s.hits == 2
        assert s.accuracy == pytest.approx(2 / 6)
        assert s.wilson_lb == pytest.approx(wilson_lower_bound(2, 6))

    def test_mean_reward_only_evaluated(self) -> None:
        rows = [
            row(rid=0, reward=2.0),
            row(rid=1, reward=4.0),
            row(rid=2, status="pending", reward=100.0),
        ]
        assert observatory_summary(rows).mean_reward == pytest.approx(3.0)

    def test_empty(self) -> None:
        s = observatory_summary([])
        assert s.total == 0 and s.evaluated == 0 and s.accuracy == 0.0


class TestDimension:
    def test_groups_by_key(self) -> None:
        rows = [
            row(rid=0, strategy="momentum", horizon=60),
            row(rid=1, strategy="momentum", horizon=60, correct=False),
            row(rid=2, strategy="naive", horizon=3600),
            row(rid=3, strategy="naive", horizon=3600, status="pending"),
        ]
        by_strategy = dimension_stats(rows, key=lambda r: r.strategy)
        momentum = next(s for s in by_strategy if s.label == "momentum")
        assert momentum.n == 2 and momentum.hits == 1
        assert momentum.predictions == 2
        naive = next(s for s in by_strategy if s.label == "naive")
        assert naive.n == 1 and naive.predictions == 2

    def test_sorted_and_empty(self) -> None:
        assert dimension_stats([], key=lambda r: r.strategy) == ()


class TestCalibration:
    def test_overconfidence_fail(self) -> None:
        rows = [
            row(rid=i, prob=0.7, correct=i < 147)
            for i in range(300)
        ]
        report = calibration_curve(rows, thresholds=CalibrationThresholds(min_n=20))
        bucket = next(b for b in report.buckets if b.label == "0.7")
        assert bucket.declared == pytest.approx(0.7)
        assert bucket.hit_rate == pytest.approx(147 / 300)
        assert bucket.status is BucketStatus.FAIL
        assert bucket.delta > 0

    def test_well_calibrated_ok(self) -> None:
        rows = [
            row(rid=i, prob=0.3, correct=i < 90)
            for i in range(300)
        ]
        report = calibration_curve(rows, thresholds=CalibrationThresholds(min_n=20))
        bucket = next(b for b in report.buckets if b.label == "0.3")
        assert bucket.status is BucketStatus.OK

    def test_insufficient(self) -> None:
        rows = [row(rid=i, prob=0.5, correct=True) for i in range(3)]
        report = calibration_curve(rows, thresholds=CalibrationThresholds(min_n=20))
        bucket = next(b for b in report.buckets if b.label == "0.5")
        assert bucket.status is BucketStatus.INSUFFICIENT

    def test_empty(self) -> None:
        report = calibration_curve([])
        assert report.buckets == ()
        assert report.ece == 0.0


class TestLearningCurve:
    def test_cumulative_accuracy_by_time(self) -> None:
        rows = [
            row(rid=i, emitted=float(i), correct=i < 12)
            for i in range(60)
        ]
        points = learning_curve(rows, targets=(20, 100, 500))
        assert [p.n for p in points] == [20, 60]
        assert points[0].accuracy == pytest.approx(12 / 20)
        assert points[1].accuracy == pytest.approx(12 / 60)
        assert points[1].wilson_lb == pytest.approx(
            wilson_lower_bound(12, 60)
        )

    def test_empty(self) -> None:
        assert learning_curve([]) == ()


class TestEvidenceRequirement:
    def test_reaches_threshold(self) -> None:
        rows = [row(rid=i, correct=i < 3000) for i in range(5000)]
        req = evidence_requirement(rows, min_accuracy=0.52)
        assert req is not None
        hits = 3000 if req >= 3000 else req
        assert wilson_lower_bound(hits, req) >= 0.52

    def test_never_reaches(self) -> None:
        rows = [row(rid=i, correct=i % 2 == 0) for i in range(5000)]
        assert evidence_requirement(rows, min_accuracy=0.52) is None

    def test_too_few_observations(self) -> None:
        rows = [row(rid=i, correct=i % 2 == 0) for i in range(10)]
        assert evidence_requirement(rows, step=20) is None

    def test_partial_final_point_counts(self) -> None:
        rows = [
            row(rid=i, correct=(i % 2 == 0) or i >= 20)
            for i in range(34)
        ]
        assert evidence_requirement(rows, min_accuracy=0.52, step=20) == 34

    def test_empty(self) -> None:
        assert evidence_requirement([]) is None

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            evidence_requirement([row()], min_accuracy=1.0)
        with pytest.raises(ValueError):
            evidence_requirement([row()], step=0)


class TestRecency:
    def test_improvement_not_degraded(self) -> None:
        rows = [
            row(rid=i, emitted=float(i), correct=i < 40)
            for i in range(100)
        ] + [
            row(rid=100 + i, emitted=1000.0 + i, correct=True)
            for i in range(300)
        ]
        bands = recency_bands(rows, bands=4)
        assert len(bands) == 4
        assert bands[-1].accuracy > bands[0].accuracy
        assert is_degraded(bands) is False

    def test_degradation_detected(self) -> None:
        rows = [
            row(rid=i, emitted=float(i), correct=True)
            for i in range(300)
        ] + [
            row(rid=300 + i, emitted=1000.0 + i, correct=i < 40)
            for i in range(100)
        ]
        bands = recency_bands(rows, bands=4)
        assert is_degraded(bands) is True

    def test_empty(self) -> None:
        assert recency_bands([]) == ()
        assert is_degraded([]) is False

    def test_invalid_bands(self) -> None:
        with pytest.raises(ValueError):
            recency_bands([row()], bands=1)


class TestRender:
    def test_dashboard_sections(self) -> None:
        rows = [row(rid=i, correct=i % 2 == 0) for i in range(40)]
        out = render_observatory(
            symbol="NVDA",
            summary=observatory_summary(rows),
            by_horizon=dimension_stats(rows, key=lambda r: f"{r.horizon_seconds}s"),
            by_strategy=dimension_stats(rows, key=lambda r: r.strategy),
            by_regime=dimension_stats(rows, key=lambda r: r.regime or "ALL"),
            calibration=calibration_curve(rows),
            contexts=[
                ContextLearning(
                    label="momentum · 3600s · TRENDING",
                    points=learning_curve(rows, targets=(20,)),
                    requirement=None,
                )
            ],
            bands=recency_bands(rows, bands=4),
            degraded=is_degraded(recency_bands(rows, bands=4)),
            evidence_min_accuracy=0.52,
        )
        for section in (
            "PREDICTION OBSERVATORY — NVDA",
            "TOTAL PREDICTIONS",
            "EVALUATED",
            "PENDING",
            "BY HORIZON",
            "BY STRATEGY",
            "BY REGIME",
            "PROBABILITY CALIBRATION",
            "LEARNING CURVE",
            "RECENCIA",
        ):
            assert section in out

    def test_render_flags_overconfidence(self) -> None:
        rows = [
            row(rid=i, prob=0.7, correct=i < 147)
            for i in range(300)
        ]
        report = calibration_curve(rows, thresholds=CalibrationThresholds(min_n=20))
        out = render_observatory(
            symbol="NVDA",
            summary=observatory_summary(rows),
            by_horizon=(),
            by_strategy=(),
            by_regime=(),
            calibration=report,
            contexts=(),
            bands=(),
            degraded=False,
            evidence_min_accuracy=0.52,
        )
        assert "SOBRECONFIANZA" in out
