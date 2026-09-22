"""Unit tests for WindowAnalyzer parameterization and pattern classification."""

import pytest
from iot_machine_learning.ml_service.runners.services.window_analyzer import WindowAnalyzer
from iot_machine_learning.ml_service.config.ml_config import OnlineBehaviorConfig
from iot_machine_learning.ml_service.sliding_window_buffer import WindowStats


def test_window_analyzer_importable():
    import iot_machine_learning.ml_service.runners.services.window_analyzer
    assert iot_machine_learning.ml_service.runners.services.window_analyzer is not None


def test_window_analyzer_parameterization_defaults():
    cfg = OnlineBehaviorConfig()
    analyzer = WindowAnalyzer(cfg)
    assert analyzer._stable_trend_threshold == 0.05
    assert analyzer._stable_var_threshold == 0.01
    assert analyzer._stable_z_score_threshold == 1.0
    assert analyzer._oscillation_var_threshold == 0.05
    assert analyzer._drifting_trend_threshold == 0.05
    assert analyzer._drifting_var_threshold == 0.05


def test_window_analyzer_custom_thresholds():
    cfg = OnlineBehaviorConfig()
    analyzer = WindowAnalyzer(
        cfg,
        stable_trend_threshold=0.10,
        stable_var_threshold=0.02,
        stable_z_score_threshold=1.5,
        oscillation_var_threshold=0.08,
        drifting_trend_threshold=0.12,
        drifting_var_threshold=0.04,
    )
    assert analyzer._stable_trend_threshold == 0.10
    assert analyzer._stable_var_threshold == 0.02
    assert analyzer._stable_z_score_threshold == 1.5
    assert analyzer._oscillation_var_threshold == 0.08
    assert analyzer._drifting_trend_threshold == 0.12
    assert analyzer._drifting_var_threshold == 0.04


def test_window_analyzer_classification_patterns():
    cfg = OnlineBehaviorConfig()
    analyzer = WindowAnalyzer(cfg)

    # STABLE: low trend, low var, low z-score
    w10 = WindowStats(window_seconds=10.0, count=10, mean=100.0, std_dev=0.005, min=99.99, max=100.01, last_value=100.0, trend=0.01)
    res = analyzer._classify_pattern(
        w1=w10,
        w5=w10,
        w10=w10,
        baseline_mean=100.0,
        z_score_last=0.2,
        is_curve_anomalous=False,
        has_microvariation=False,
    )
    assert res == "STABLE"

    # OSCILLATING: high var, sign changes
    w1 = WindowStats(window_seconds=1.0, count=1, mean=100.0, std_dev=0.1, min=99.0, max=101.0, last_value=100.0, trend=0.05)
    w5 = WindowStats(window_seconds=5.0, count=5, mean=100.0, std_dev=0.1, min=99.0, max=101.0, last_value=100.0, trend=-0.05)
    w10_osc = WindowStats(window_seconds=10.0, count=10, mean=100.0, std_dev=0.08, min=98.0, max=102.0, last_value=100.0, trend=0.0)
    res = analyzer._classify_pattern(
        w1=w1,
        w5=w5,
        w10=w10_osc,
        baseline_mean=100.0,
        z_score_last=0.5,
        is_curve_anomalous=False,
        has_microvariation=False,
    )
    assert res == "OSCILLATING"

    # DRIFTING: high trend, low var
    w10_drift = WindowStats(window_seconds=10.0, count=10, mean=100.0, std_dev=0.02, min=99.0, max=101.0, last_value=101.0, trend=0.08)
    res = analyzer._classify_pattern(
        w1=w10_drift,
        w5=w10_drift,
        w10=w10_drift,
        baseline_mean=100.0,
        z_score_last=0.8,
        is_curve_anomalous=False,
        has_microvariation=False,
    )
    assert res == "DRIFTING"
