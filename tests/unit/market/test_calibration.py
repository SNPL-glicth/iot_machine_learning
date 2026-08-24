"""Calibración (FASE 7.5) — tests del módulo puro.

Bucket status (OK/INSUFFICIENT/FAIL), ECE ponderado, curva ASCII y
render del tablero (función pura del script, sin MySQL).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from iot_machine_learning.domain.entities.market.replay.calibration import (
    BucketStatus,
    CalibrationThresholds,
    bucket_calibration,
    calibration_chart,
)

_SCRIPT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "scripts"
    / "zenin_dashboard.py"
)


def _load_dashboard():
    spec = importlib.util.spec_from_file_location("zenin_dashboard_mod", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["zenin_dashboard_mod"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def dashboard():
    return _load_dashboard()


class TestBucketCalibration:
    def test_ok_when_within_tolerance(self) -> None:
        report = bucket_calibration([("0.5", 0.51, 5, 10)])
        bucket = report.buckets[0]
        assert bucket.status is BucketStatus.OK
        assert bucket.hit_rate == pytest.approx(0.5)
        assert bucket.delta == pytest.approx(0.01)

    def test_fail_when_off_by_more_than_tolerance(self) -> None:
        report = bucket_calibration([("0.9", 0.90, 1, 10)])
        bucket = report.buckets[0]
        assert bucket.status is BucketStatus.FAIL
        assert bucket.hit_rate == pytest.approx(0.1)
        assert bucket.delta == pytest.approx(0.8)
        assert report.has_failures

    def test_insufficient_when_sample_small(self) -> None:
        report = bucket_calibration([("0.9", 0.90, 0, 3)])
        bucket = report.buckets[0]
        assert bucket.status is BucketStatus.INSUFFICIENT
        assert not report.has_failures
        assert report.insufficient_buckets == (bucket,)

    def test_insufficient_needs_min_n_samples(self) -> None:
        """n == min_n concluye; n < min_n no (guardrail de FASE 8)."""
        ok = bucket_calibration([("0.7", 0.70, 7, 10)], CalibrationThresholds(min_n=5))
        assert ok.buckets[0].status is BucketStatus.OK
        small = bucket_calibration([("0.7", 0.70, 3, 4)], CalibrationThresholds(min_n=5))
        assert small.buckets[0].status is BucketStatus.INSUFFICIENT

    def test_failure_zero_hits_large_sample(self) -> None:
        """El caso del dashboard: P=0.9 con 0/6 → FAIL (no maquillado)."""
        report = bucket_calibration([("0.9", 0.90, 0, 6)])
        assert report.buckets[0].status is BucketStatus.FAIL

    def test_ece_weighted_by_sample(self) -> None:
        report = bucket_calibration(
            [
                ("0.5", 0.50, 5, 10),  # delta 0.0
                ("0.9", 0.90, 0, 10),  # delta 0.9, peso 10/20
            ]
        )
        assert report.ece == pytest.approx(0.9 * 0.5)

    def test_empty_report(self) -> None:
        report = bucket_calibration([])
        assert report.ece == 0.0
        assert report.buckets == ()
        assert not report.has_failures

    def test_invalid_sample_rejected(self) -> None:
        with pytest.raises(ValueError, match="muestra inválida"):
            bucket_calibration([("0.5", 0.5, 11, 10)])
        with pytest.raises(ValueError, match="declared"):
            bucket_calibration([("0.5", 1.5, 1, 10)])

    def test_invalid_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_n"):
            CalibrationThresholds(min_n=0)
        with pytest.raises(ValueError, match="tolerance"):
            CalibrationThresholds(tolerance=1.5)


class TestCalibrationChart:
    def test_chart_contains_markers_and_diagonal(self) -> None:
        report = bucket_calibration(
            [("0.5", 0.50, 5, 10), ("0.9", 0.90, 0, 6)]
        )
        chart = calibration_chart(report)
        assert "o" in chart
        assert "x" in chart
        assert "\\" in chart
        assert "declarado P(up)" in chart

    def test_chart_without_data(self) -> None:
        chart = calibration_chart(bucket_calibration([]))
        assert "sin datos" in chart

    def test_chart_too_small_rejected(self) -> None:
        report = bucket_calibration([("0.5", 0.5, 5, 10)])
        with pytest.raises(ValueError, match="curva demasiado pequeña"):
            calibration_chart(report, width=5, height=3)


class TestDashboardRender:
    def test_human_horizon(self, dashboard) -> None:
        assert dashboard.human_horizon(60) == "1m"
        assert dashboard.human_horizon(300) == "5m"
        assert dashboard.human_horizon(900) == "15m"
        assert dashboard.human_horizon(3600) == "1h"
        assert dashboard.human_horizon(999) == "999s"

    def test_render_contains_all_sections(self, dashboard) -> None:
        stats = {
            "predictions": 156,
            "evaluated": 96,
            "pending": 0,
            "invalidated": 60,
            "hits": 64,
            "brier": 0.547,
            "reward": 55.1027,
        }
        history = {
            "by_horizon": [
                {"key": 60, "evaluated": 38, "hits": 23, "reward": 15.0},
                {"key": 300, "evaluated": 34, "hits": 24, "reward": 22.5},
            ],
            "by_regime": [{"key": None, "evaluated": 96, "hits": 64}],
            "by_confidence": [
                {"bucket": 0.5, "evaluated": 6, "hits": 3, "avg_probability": 0.505},
                {"bucket": 0.9, "evaluated": 6, "hits": 0, "avg_probability": 0.896},
            ],
        }
        report = bucket_calibration(
            [
                ("0.5", 0.505, 3, 6),
                ("0.9", 0.896, 0, 6),
            ]
        )
        rendered = dashboard.render_dashboard(stats, history, report, symbol="NVDA")
        assert "ZENIN MARKET" in rendered
        assert "Predictions" in rendered
        assert "HORIZON" in rendered
        assert "1m" in rendered and "5m" in rendered
        assert "CONFIDENCE" in rendered
        assert "⚠ CALIBRATION FAILURE" in rendered
        assert "REGIME" in rendered
        assert "CALIBRATION CURVE" in rendered
        assert "ECE" in rendered and "Brier" in rendered

    def test_render_honest_insufficient(self, dashboard) -> None:
        """Buckets con n < min_n se marcan y NO se aprenden (guardrail)."""
        stats = {k: 0 for k in ("predictions", "evaluated", "pending",
                                "invalidated", "hits", "brier", "reward")}
        history = {"by_horizon": [], "by_regime": [], "by_confidence": []}
        report = bucket_calibration([("0.6", 0.58, 0, 3)])
        rendered = dashboard.render_dashboard(stats, history, report)
        assert "⚠ insufficient" in rendered
        assert "no concluye (guardrail FASE 8" in rendered

    def test_all_rows_same_width(self, dashboard) -> None:
        stats = {
            "predictions": 156,
            "evaluated": 96,
            "pending": 0,
            "invalidated": 60,
            "hits": 64,
            "brier": 0.547,
            "reward": 55.1027,
        }
        history = {
            "by_horizon": [{"key": 60, "evaluated": 38, "hits": 23, "reward": 15.0}],
            "by_regime": [],
            "by_confidence": [],
        }
        report = bucket_calibration([])
        rendered = dashboard.render_dashboard(stats, history, report)
        lines = [
            ln for ln in rendered.splitlines() if ln.startswith("║")
        ]
        widths = {len(ln) for ln in lines}
        assert len(widths) == 1, f"box rows with mismatched widths: {widths}"
