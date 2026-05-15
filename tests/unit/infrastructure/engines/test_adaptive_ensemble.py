"""Tests para AdaptiveEnsembleEngine (P6).

Verifica:
1. Regime detection heurística: noisy / trending / stable
2. Engine selection según regime
3. record_actual() se propaga a todos los sub-engines
4. Fallback cuando el engine preferido no puede manejar los datos
"""

from __future__ import annotations

import random

import pytest

from iot_machine_learning.infrastructure.ml.engines.adaptive_ensemble import (
    AdaptiveEnsembleEngine,
)


class TestAdaptiveEnsembleRegimeDetection:
    """P6: heuristic regime detector."""

    def test_noisy_regime_selects_baseline(self) -> None:
        """Ruido gaussiano puro → regime=noisy → baseline (benchmark evidence)."""
        random.seed(42)
        engine = AdaptiveEnsembleEngine()
        values = [random.gauss(50, 20) for _ in range(100)]
        result = engine.predict(values)

        assert result.metadata["ensemble_regime"] == "noisy"
        assert result.metadata["ensemble_selected"] == "baseline"

    def test_volatile_regime_selects_kalman(self) -> None:
        """Ruido con dinámica subyacente → regime=volatile → kalman."""
        random.seed(42)
        engine = AdaptiveEnsembleEngine(volatile_threshold=0.4)
        # High variance relative to mean → volatile regime (has underlying dynamics)
        values = [random.gauss(10, 15) for _ in range(100)]
        result = engine.predict(values)

        assert result.metadata["ensemble_regime"] == "volatile"
        assert result.metadata["ensemble_selected"] == "kalman"

    def test_trending_regime_selects_taylor(self) -> None:
        """Salto pronunciado al final → regime=trending → taylor."""
        engine = AdaptiveEnsembleEngine()
        # 90% estable + 10% con pendiente fuerte: low global std, high local slope
        values = [10.0] * 90 + [10.0 + i * 0.5 for i in range(10)]
        result = engine.predict(values)

        assert result.metadata["ensemble_regime"] == "trending"
        assert result.metadata["ensemble_selected"] == "taylor"

    def test_stable_regime_selects_baseline(self) -> None:
        """Señal estable → regime=stable → baseline."""
        engine = AdaptiveEnsembleEngine()
        values = [50.0 + random.gauss(0, 0.5) for _ in range(100)]
        result = engine.predict(values)

        assert result.metadata["ensemble_regime"] == "stable"
        assert result.metadata["ensemble_selected"] == "baseline"


class TestAdaptiveEnsembleFallback:
    """P6: fallback chain when preferred engine cannot handle data."""

    def test_taylor_fallback_to_baseline_with_few_points(self) -> None:
        """Trending regime pero pocos puntos → fallback a baseline."""
        engine = AdaptiveEnsembleEngine()
        # 2 puntos estables + 1 punto con salto: trending pero too few for taylor
        values = [10.0, 10.0, 15.0]  # trending but too few for taylor (needs 4)
        result = engine.predict(values)

        assert result.metadata["ensemble_regime"] == "trending"
        assert result.metadata["ensemble_selected"] == "baseline"

    def test_can_handle_is_or_of_subengines(self) -> None:
        """can_handle returns True if ANY sub-engine can handle."""
        engine = AdaptiveEnsembleEngine()
        assert engine.can_handle(1) is True   # baseline
        assert engine.can_handle(3) is True   # statistical
        assert engine.can_handle(4) is True   # taylor


class TestAdaptiveEnsembleRecordActual:
    """P6: record_actual propagates to all sub-engines."""

    def test_record_actual_to_all_engines(self) -> None:
        """Each sub-engine receives the actual value."""
        engine = AdaptiveEnsembleEngine()
        engine.record_actual(10.0, 12.0)

        # Baseline (always has deque)
        assert len(engine._baseline._error_history) == 1

        # Taylor (has tracker)
        if engine._taylor._tracker:
            assert engine._taylor._tracker._errors  # type: ignore[attr-defined]

        # Statistical (has history)
        assert len(engine._statistical._prediction_history) == 1

        # Kalman (has deque)
        assert len(engine._kalman._error_history) == 1


class TestAdaptiveEnsembleInterface:
    """P6: PredictionEngine contract."""

    def test_name(self) -> None:
        """Engine name is adaptive_ensemble."""
        engine = AdaptiveEnsembleEngine()
        assert engine.name == "adaptive_ensemble"

    def test_supports_uncertainty(self) -> None:
        """Adaptive ensemble does not provide intervals."""
        engine = AdaptiveEnsembleEngine()
        assert engine.supports_uncertainty() is False

    def test_empty_values_raises(self) -> None:
        """Empty values must raise ValueError."""
        engine = AdaptiveEnsembleEngine()
        with pytest.raises(ValueError):
            engine.predict([])
