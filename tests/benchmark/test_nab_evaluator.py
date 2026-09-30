"""Tests for NABEvaluator (Fase 1: Evaluation Harness)."""

from __future__ import annotations

import pytest
import numpy as np

from benchmarks.nab_evaluator import (
    AnomalyWindow,
    NABEvaluator,
    ComprehensiveNABReport,
)


def test_window_construction():
    evaluator = NABEvaluator(window_size_points=5)
    series_ts = [float(i * 10) for i in range(100)]
    anomaly_ts = [250.0, 750.0]

    windows = evaluator.build_windows_from_timestamps(series_ts, anomaly_ts)
    assert len(windows) == 2

    # Primer evento en index 25
    assert windows[0].start_idx == 20
    assert windows[0].end_idx == 30
    assert windows[0].center_time == 250.0

    # Segundo evento en index 75
    assert windows[1].start_idx == 70
    assert windows[1].end_idx == 80
    assert windows[1].center_time == 750.0


def test_perfect_detector():
    evaluator = NABEvaluator(window_size_points=5)
    n = 100
    series_ts = [float(i) for i in range(n)]
    anomaly_ts = [20.0, 60.0]
    windows = evaluator.build_windows_from_timestamps(series_ts, anomaly_ts)

    # Detector perfecto: predice exactamente en el inicio de cada ventana
    preds = [0] * n
    for w in windows:
        preds[w.start_idx] = 1

    report = evaluator.evaluate(preds, scores=None, series_timestamps=series_ts, anomaly_timestamps=anomaly_ts)

    # Event Recall debe ser 100%
    assert report.event_level.recall_event == 1.0
    assert report.event_level.tp_events == 2
    assert report.event_level.fn_events == 0
    assert report.event_level.fp_points == 0
    assert report.event_level.f1_event_cluster == 1.0

    # Score de NAB debe ser 100.0
    assert report.nab_scoring.standard_score == pytest.approx(100.0, rel=1e-2)
    assert report.nab_scoring.low_fp_score == pytest.approx(100.0, rel=1e-2)
    assert report.nab_scoring.low_fn_score == pytest.approx(100.0, rel=1e-2)


def test_null_detector():
    evaluator = NABEvaluator(window_size_points=5)
    n = 100
    series_ts = [float(i) for i in range(n)]
    anomaly_ts = [20.0, 60.0]

    preds = [0] * n
    report = evaluator.evaluate(preds, scores=None, series_timestamps=series_ts, anomaly_timestamps=anomaly_ts)

    assert report.event_level.recall_event == 0.0
    assert report.event_level.tp_events == 0
    assert report.event_level.fn_events == 2
    assert report.event_level.fp_points == 0

    # Score de NAB para null detector se normaliza exactamente a 0.0
    assert report.nab_scoring.standard_score == pytest.approx(0.0, abs=1e-3)


def test_false_positive_clusters():
    evaluator = NABEvaluator(window_size_points=5)
    n = 100
    series_ts = [float(i) for i in range(n)]
    anomaly_ts = [50.0]  # Ventana [45, 55]

    preds = [0] * n
    # TP en el evento
    preds[48] = 1

    # Cluster 1 de FP: puntos 5, 6, 7 (3 puntos continuos = 1 cluster)
    preds[5] = preds[6] = preds[7] = 1
    # Cluster 2 de FP: punto 80 (1 punto = 1 cluster)
    preds[80] = 1

    report = evaluator.evaluate(preds, scores=None, series_timestamps=series_ts, anomaly_timestamps=anomaly_ts)

    assert report.event_level.tp_events == 1
    assert report.event_level.fp_points == 4
    assert report.event_level.fp_clusters == 2
    # Precision de cluster: 1 TP / (1 TP + 2 clusters) = 1/3
    assert report.event_level.precision_event_cluster == pytest.approx(1.0 / 3.0)


def test_summary_markdown_generation():
    evaluator = NABEvaluator(window_size_points=5)
    n = 50
    series_ts = [float(i) for i in range(n)]
    anomaly_ts = [25.0]
    preds = [0] * n
    preds[25] = 1

    report = evaluator.evaluate(preds, scores=[float(p) for p in preds], series_timestamps=series_ts, anomaly_timestamps=anomaly_ts, dataset_name="Test Unit")
    md = report.summary_markdown()

    assert "Test Unit" in md
    assert "Event Recall" in md
    assert "NAB Standard Profile" in md
    assert "Range F1" in md
