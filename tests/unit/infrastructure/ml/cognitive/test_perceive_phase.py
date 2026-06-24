"""Tests for PerceivePhase — seasonal strength, hysteresis, cross-regime coherence."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from iot_machine_learning.domain.entities.series.structural_analysis import (
    RegimeType,
    StructuralAnalysis,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import (
    PipelineContext,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.perceive_phase import (
    PerceivePhase,
)


# ── helpers ──────────────────────────────────────────────────────────


def _make_ctx(
    *,
    orchestrator=None,
    values=None,
    series_id="test_sensor",
    **overrides,
) -> PipelineContext:
    if orchestrator is None:
        orchestrator = MagicMock()
        orchestrator._analyzer = MagicMock()
        orchestrator._correlation_port = None
        orchestrator._series_values_store = None
        orchestrator._context_state_manager = None
        orchestrator._enable_advanced_plasticity = False
        orchestrator._plasticity_coordinator = None
        orchestrator._sensor_profile_repository = None
        orchestrator._metrics_collector = None
    if values is None:
        values = [float(v) for v in range(20)]
    return PipelineContext(
        orchestrator=orchestrator,
        values=values,
        timestamps=list(range(len(values))),
        series_id=series_id,
        flags=MagicMock(),
        timer=MagicMock(),
        **overrides,
    )


def _make_analysis(
    *,
    regime: RegimeType = RegimeType.STABLE,
    mean: float = 100.0,
    std: float = 1.0,
    slope: float = 0.0,
    noise_ratio: float = 0.01,
    stability: float = 0.1,
    curvature: float = 0.0,
) -> StructuralAnalysis:
    return StructuralAnalysis(
        slope=slope,
        curvature=curvature,
        stability=stability,
        noise_ratio=noise_ratio,
        regime=regime,
        mean=mean,
        std=std,
        trend_strength=abs(slope) / abs(mean) if abs(mean) > 1e-9 else 0.0,
        n_points=20,
    )


# ── test seasonal_strength > 0.5 ──────────────────────────────────────


class TestSeasonalStrength:
    """Seasonal decomposition produces positive seasonal_strength."""

    def test_seasonal_strength_above_half(self):
        """Seasonal values produce seasonal_strength > 0.5."""
        n = 24
        period = 6
        base = 100.0
        seasonal_amp = 10.0
        seasonal = [
            seasonal_amp * math.sin(2 * math.pi * (i % period) / period)
            for i in range(n)
        ]
        values = [base + s for s in seasonal]

        phase = PerceivePhase(seasonal_min_points=6)
        strength, dom_period, trend, residual = phase._decompose(values, "s1")
        assert strength > 0.5, f"expected seasonal_strength > 0.5, got {strength}"
        assert dom_period > 0, f"expected dominant_period > 0, got {dom_period}"

    def test_flat_signal_returns_zero_strength(self):
        """Flat signal yields seasonal_strength ~ 0.0."""
        values = [100.0] * 20
        phase = PerceivePhase(seasonal_min_points=6)
        strength, dom_period, trend, residual = phase._decompose(values, "s1")
        assert strength < 0.1, f"expected strength near 0, got {strength}"
        assert dom_period == 0

    def test_short_signal_returns_zero(self):
        """Too few points → no decomposition."""
        values = [1.0, 2.0, 3.0]
        phase = PerceivePhase(seasonal_min_points=10)
        strength, dom_period, trend, residual = phase._decompose(values, "s1")
        assert strength == 0.0
        assert dom_period == 0
        assert trend is None

    def test_seasonal_strength_propagates_to_feature_context(self):
        """After execute(), feature_context.seasonal_strength > 0."""
        n = 24
        period = 6
        values = [
            100.0 + 10.0 * math.sin(2 * math.pi * (i % period) / period)
            for i in range(n)
        ]

        orchestrator = MagicMock()
        orchestrator._analyzer.analyze.return_value = _make_analysis()
        orchestrator._correlation_port = None
        orchestrator._series_values_store = None
        orchestrator._context_state_manager = None
        orchestrator._enable_advanced_plasticity = False
        orchestrator._plasticity_coordinator = None
        orchestrator._sensor_profile_repository = None
        orchestrator._metrics_collector = None

        ctx = _make_ctx(orchestrator=orchestrator, values=values, series_id="s1")
        phase = PerceivePhase(seasonal_min_points=6)
        result = phase.execute(ctx)

        assert result.feature_context is not None
        assert result.feature_context.seasonal_strength > 0.5
        assert result.feature_context.dominant_period > 0


# ── test hysteresis prevents oscillation ──────────────────────────────


class TestHysteresis:
    """Regime hysteresis prevents rapid flips on oscillating input."""

    def _make_phase(self, hysteresis_n=2):
        return PerceivePhase(hysteresis_n=hysteresis_n)

    def test_initial_call_establishes_baseline(self):
        """First call with a new series_id confirms the regime directly."""
        phase = self._make_phase()
        confirmed, conf, stability = phase._resolve_regime_with_hysteresis(
            "s_hyst", "STABLE", redis_client=None,
        )
        assert confirmed == "STABLE"
        assert conf >= 0.8

    def test_single_glitch_does_not_flip(self):
        """A single different reading does not flip the confirmed regime."""
        phase = self._make_phase(hysteresis_n=2)

        confirmed, _, _ = phase._resolve_regime_with_hysteresis(
            "s_glitch", "STABLE", redis_client=None,
        )
        assert confirmed == "STABLE"

        confirmed, _, _ = phase._resolve_regime_with_hysteresis(
            "s_glitch", "VOLATILE", redis_client=None,
        )
        assert confirmed == "STABLE"

    def test_repeated_challenger_eventually_flips(self):
        """After N consecutive challenger readings regime flips."""
        phase = self._make_phase(hysteresis_n=3)
        sid = "s_flip"

        phase._resolve_regime_with_hysteresis(sid, "STABLE", redis_client=None)
        phase._resolve_regime_with_hysteresis(sid, "STABLE", redis_client=None)

        for _ in range(2):
            confirmed, _, _ = phase._resolve_regime_with_hysteresis(
                sid, "VOLATILE", redis_client=None,
            )
            assert confirmed == "STABLE"

        confirmed, _, _ = phase._resolve_regime_with_hysteresis(
            sid, "VOLATILE", redis_client=None,
        )
        assert confirmed == "VOLATILE"

    def test_hysteresis_counter_resets_on_return(self):
        """Counter resets when regime returns to confirmed before threshold."""
        phase = self._make_phase(hysteresis_n=3)
        sid = "s_reset"

        phase._resolve_regime_with_hysteresis(sid, "STABLE", redis_client=None)

        phase._resolve_regime_with_hysteresis(sid, "VOLATILE", redis_client=None)
        phase._resolve_regime_with_hysteresis(sid, "VOLATILE", redis_client=None)

        confirmed, _, _ = phase._resolve_regime_with_hysteresis(
            sid, "STABLE", redis_client=None,
        )
        assert confirmed == "STABLE"

        # Now start fresh — VOLATILE needs 3 consecutive again
        for _ in range(2):
            confirmed, _, _ = phase._resolve_regime_with_hysteresis(
                sid, "VOLATILE", redis_client=None,
            )
            assert confirmed == "STABLE"

    def test_confidence_increases_with_counter(self):
        """Confidence during challenger phase increases gradually."""
        phase = self._make_phase(hysteresis_n=4)
        sid = "s_conf"

        phase._resolve_regime_with_hysteresis(sid, "STABLE", redis_client=None)
        phase._resolve_regime_with_hysteresis(sid, "STABLE", redis_client=None)

        confs = []
        for _ in range(3):
            _, conf, _ = phase._resolve_regime_with_hysteresis(
                sid, "VOLATILE", redis_client=None,
            )
            confs.append(conf)
        # confidence should be monotonically increasing
        assert confs == sorted(confs), f"confidence not increasing: {confs}"

    def test_different_series_are_independent(self):
        """Hysteresis state for series A does not affect series B."""
        phase = self._make_phase(hysteresis_n=2)

        phase._resolve_regime_with_hysteresis("s_a", "STABLE", redis_client=None)
        phase._resolve_regime_with_hysteresis("s_a", "VOLATILE", redis_client=None)
        confirmed_a, _, _ = phase._resolve_regime_with_hysteresis(
            "s_a", "VOLATILE", redis_client=None,
        )

        confirmed_b, _, _ = phase._resolve_regime_with_hysteresis(
            "s_b", "VOLATILE", redis_client=None,
        )

        assert confirmed_a == "VOLATILE"
        assert confirmed_b == "VOLATILE"


# ── test cross-regime coherence ──────────────────────────────────────


class TestCrossRegimeCoherence:
    """Cross-regime coherence detection works correctly."""

    @staticmethod
    def _check(regime: str, neighbors: dict) -> bool:
        return PerceivePhase._check_cross_regime_coherence(regime, neighbors)

    def test_volatile_with_stable_is_incoherent(self):
        """VOLATILE when neighbors are STABLE is incoherent."""
        assert self._check("VOLATILE", {"n1": "STABLE", "n2": "STABLE"})

    def test_stable_with_volatile_is_incoherent(self):
        """STABLE when neighbors are VOLATILE is incoherent."""
        assert self._check("STABLE", {"n1": "VOLATILE"})

    def test_same_regime_is_coherent(self):
        """All STABLE is coherent."""
        assert not self._check("STABLE", {"n1": "STABLE", "n2": "STABLE"})

    def test_volatile_with_volatile_is_coherent(self):
        """All VOLATILE is coherent."""
        assert not self._check("VOLATILE", {"n1": "VOLATILE"})

    def test_trending_with_stable_is_incoherent(self):
        """TRENDING when neighbor is STABLE is incoherent."""
        assert self._check("TRENDING", {"n1": "STABLE"})

    def test_unknown_regime_is_always_coherent(self):
        """UNKNOWN does not trigger incoherence."""
        assert not self._check("UNKNOWN", {"n1": "STABLE"})
        assert not self._check("UNKNOWN", {"n1": "VOLATILE"})

    def test_mixed_neighbors_picks_incoherence(self):
        """One VOLATILE neighbor among STABLE still flags incoherence."""
        assert self._check("STABLE", {"n1": "STABLE", "n2": "VOLATILE"})

    def test_trending_with_trending_is_coherent(self):
        """All TRENDING is coherent."""
        assert not self._check("TRENDING", {"n1": "TRENDING"})

    def test_noisy_grouped_with_volatile(self):
        """NOISY groups with VOLATILE in the volatile group."""
        assert self._check("NOISY", {"n1": "STABLE"})
        assert not self._check("NOISY", {"n1": "VOLATILE"})

    def test_empty_neighbors_is_coherent(self):
        """No neighbors means no incoherence."""
        assert not self._check("STABLE", {})
        assert not self._check("VOLATILE", {})


# ── test execute() end-to-end with mocks ──────────────────────────────


class TestPerceivePhaseExecute:
    """End-to-end execute() with wired mocks."""

    @pytest.fixture
    def phase(self):
        return PerceivePhase(seasonal_min_points=6, hysteresis_n=2)

    @pytest.fixture
    def bare_orchestrator(self):
        orch = MagicMock()
        orch._analyzer = MagicMock()
        orch._analyzer.analyze.return_value = _make_analysis()
        orch._correlation_port = None
        orch._series_values_store = None
        orch._context_state_manager = None
        orch._enable_advanced_plasticity = False
        orch._plasticity_coordinator = None
        orch._sensor_profile_repository = None
        orch._metrics_collector = None
        return orch

    def test_execute_sets_regime_and_feature_context(self, phase, bare_orchestrator):
        """After execute, regime and feature_context are populated."""
        values = [float(v) for v in range(20)]
        ctx = _make_ctx(orchestrator=bare_orchestrator, values=values)
        result = phase.execute(ctx)
        assert result.regime is not None
        assert result.feature_context is not None
        assert result.feature_context.regime == result.regime

    def test_execute_sets_regime_confidence(self, phase, bare_orchestrator):
        """After execute, regime_confidence is populated on context."""
        values = [float(v) for v in range(20)]
        ctx = _make_ctx(orchestrator=bare_orchestrator, values=values)
        result = phase.execute(ctx)
        assert 0.0 <= result.regime_confidence <= 1.0
        assert result.regime_confidence == pytest.approx(
            result.feature_context.regime_confidence,
        )

    def test_cross_regime_incoherence_set(self, phase, bare_orchestrator):
        """cross_regime_incoherence is set on FeatureContext."""
        values = [float(v) for v in range(20)]
        ctx = _make_ctx(orchestrator=bare_orchestrator, values=values)
        result = phase.execute(ctx)
        assert isinstance(result.cross_regime_incoherence, bool)
        assert result.cross_regime_incoherence == (
            result.feature_context.cross_regime_incoherence
        )
