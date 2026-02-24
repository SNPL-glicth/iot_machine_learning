"""Tests for cognitive/signal_analyzer.py — StructuralAnalysis extraction."""

from __future__ import annotations

import math
import random

import pytest

from iot_machine_learning.domain.entities.series.structural_analysis import (
    RegimeType,
    StructuralAnalysis,
)
from iot_machine_learning.infrastructure.ml.cognitive.analysis.signal_analyzer import (
    SignalAnalyzer,
)


class TestSignalAnalyzerBasic:

    def test_empty_values(self) -> None:
        a = SignalAnalyzer()
        p = a.analyze([])
        assert isinstance(p, StructuralAnalysis)
        assert p.n_points == 0
        assert p.mean == 0.0
        assert p.regime == RegimeType.STABLE

    def test_constant_signal(self) -> None:
        a = SignalAnalyzer()
        p = a.analyze([5.0] * 20)
        assert p.n_points == 20
        assert p.mean == pytest.approx(5.0)
        assert p.std == pytest.approx(0.0)
        assert p.noise_ratio == pytest.approx(0.0)
        assert p.slope == pytest.approx(0.0, abs=1e-10)
        assert p.regime == RegimeType.STABLE

    def test_linear_signal(self) -> None:
        a = SignalAnalyzer()
        # High baseline so noise_ratio stays low (std/mean << 0.5)
        values = [100.0 + float(i) for i in range(20)]
        p = a.analyze(values)
        assert p.slope == pytest.approx(1.0, abs=0.01)
        assert p.regime == RegimeType.TRENDING

    def test_noisy_signal(self) -> None:
        random.seed(42)
        a = SignalAnalyzer()
        values = [10.0 + random.gauss(0, 10.0) for _ in range(50)]
        p = a.analyze(values)
        assert p.noise_ratio > 0.3
        assert p.regime == RegimeType.NOISY

    def test_curvature_quadratic(self) -> None:
        a = SignalAnalyzer()
        values = [float(i * i) for i in range(20)]
        p = a.analyze(values)
        assert p.curvature == pytest.approx(2.0, abs=0.01)

    def test_timestamps_affect_dt(self) -> None:
        a = SignalAnalyzer()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ts = [0.0, 10.0, 20.0, 30.0, 40.0]
        p = a.analyze(values, ts)
        assert p.dt == pytest.approx(10.0)

    def test_to_dict_keys(self) -> None:
        a = SignalAnalyzer()
        p = a.analyze([1.0, 2.0, 3.0])
        d = p.to_dict()
        expected = {"n_points", "mean", "std", "noise_ratio",
                    "slope", "curvature", "regime", "dt",
                    "stability", "accel_variance", "trend_strength"}
        assert set(d.keys()) == expected


class TestRegimeClassification:

    def test_stable_regime(self) -> None:
        a = SignalAnalyzer()
        p = a.analyze([100.0 + i * 0.001 for i in range(50)])
        assert p.regime == RegimeType.STABLE

    def test_trending_regime(self) -> None:
        a = SignalAnalyzer()
        # High baseline so noise_ratio stays low
        p = a.analyze([500.0 + float(i * 5) for i in range(50)])
        assert p.regime == RegimeType.TRENDING

    def test_volatile_regime(self) -> None:
        random.seed(99)
        a = SignalAnalyzer()
        # σ=25 on mean≈100 → noise_ratio ≈ 0.25 → volatile
        values = [100.0 + random.gauss(0, 25.0) for _ in range(50)]
        p = a.analyze(values)
        assert p.regime in (RegimeType.VOLATILE, RegimeType.NOISY)
