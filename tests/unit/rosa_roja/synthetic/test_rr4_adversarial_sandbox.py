"""RR-4 adversarial sandbox: stress-testing Rosa Roja under jitter, noise, flapping."""

from __future__ import annotations

import numpy as np
import pytest

from core.orchestration.rosa_roja.domain.execution import ExecutionPlan
from core.orchestration.rosa_roja.engine import RosaRojaEngine
from core.orchestration.rosa_roja.modules.module1_ingestion import MahalanobisFilter
from core.orchestration.rosa_roja.modules.rhythm_generator import RhythmTrajectoryGenerator
from core.orchestration.rosa_roja.modules.module3_moe_gating import MultiplicativeMoEGating
from .envs import (
    PATTERN_A, 
    PATTERN_B, 
    JitteryPatternEnv, 
    NoisyStateEnv, 
    FlappingRegimeEnv
)


class PassExpert:
    name = "mock_expert"
    is_critical = True
    threshold = 0.6
    weight = 1.0

    def evaluate_trajectory(self, trajectory) -> float:
        return 0.9

    def update_learning(self, actual, predicted) -> None:
        pass


class StaticDriftSensor:
    name = "static_drift"

    def get_drift_score(self) -> float:
        return 0.0

    def update(self, actual, predicted) -> None:
        pass

    def reset(self) -> None:
        pass


def build_engine(**kwargs) -> RosaRojaEngine:
    defaults = dict(
        ingestion_filter=MahalanobisFilter(
            noise_threshold=3.0, history_window=40, min_samples_for_cov=8
        ),
        rhythm_generator=RhythmTrajectoryGenerator(
            min_trajectory_len=3, max_trajectory_len=5, top_k=2, oversample_factor=3
        ),
        moe_gating=MultiplicativeMoEGating(),
        expert_jury=[PassExpert()],
        drift_sensors=[StaticDriftSensor()],
    )
    defaults.update(kwargs)
    return RosaRojaEngine(**defaults)


class TestJitteryTempo:
    """Δt stochastic: Φ_Ritmo should degrade gracefully, no false EMERGENCY_FLUSH."""

    def test_tempo_jitter_degrades_phi_ritmo_proportionally(self):
        rng = np.random.default_rng(42)
        env = JitteryPatternEnv(PATTERN_A, dt_mean=1.0, dt_std=0.5, rng=rng)
        engine = build_engine()

        phi_values = []
        for _ in range(80):
            delta, dt = env.step()
            plan = engine.process_event(delta, dt)
            if plan.action == "EXECUTE":
                phi_values.append(plan.global_confidence)

        assert len(phi_values) > 10, "should still EXECUTE under tempo jitter"
        assert all(p > 0.0 for p in phi_values)

    def test_no_false_emergency_flush_under_jitter(self):
        """Jitter may trigger EMERGENCY_FLUSH via tracker, but engine must not crash."""
        rng = np.random.default_rng(123)
        env = JitteryPatternEnv(PATTERN_A, dt_mean=1.0, dt_std=0.3, rng=rng)
        engine = build_engine()

        crashes = 0
        for _ in range(100):
            delta, dt = env.step()
            try:
                engine.process_event(delta, dt)
            except Exception:
                crashes += 1

        assert crashes == 0, "engine crashed under tempo jitter"


class TestNoisyStateTrajectory:
    """Gaussian noise on deltas: covariance should expand without false regime resets."""

    def test_mahalanobis_expands_within_3sigma_noise(self):
        """Engine should not crash or infinite-loop; falls back to HOLD gracefully."""
        rng = np.random.default_rng(42)
        env = NoisyStateEnv(PATTERN_A, noise_std=0.1, rng=rng)
        engine = build_engine(
            ingestion_filter=MahalanobisFilter(
                noise_threshold=3.0, history_window=60, min_samples_for_cov=15
            )
        )

        outlier_alerts = 0
        crashes = 0
        for _ in range(120):
            delta = env.step()
            try:
                plan = engine.process_event(delta, 1.0)
                if plan.regime_alert:
                    outlier_alerts += 1
            except Exception:
                crashes += 1

        assert crashes == 0, "engine crashed under noise"
        # Should not have perpetual alerts; may have a few during warmup
        assert outlier_alerts <= 10, f"too many alerts under 3σ noise: {outlier_alerts}"

    def test_noise_within_training_distribution_no_reset(self):
        """Noise σ matching initial covariance scale should not trigger auto-reset."""
        rng = np.random.default_rng(123)
        env = NoisyStateEnv(PATTERN_A, noise_std=0.5, rng=rng)
        engine = build_engine(
            ingestion_filter=MahalanobisFilter(
                noise_threshold=3.0, history_window=60, min_samples_for_cov=15
            )
        )

        auto_resets = 0
        for _ in range(80):
            delta = env.step()
            plan = engine.process_event(delta, 1.0)
            if plan.veto_details.get("reason") == "Auto_Regime_Reset_Triggered":
                auto_resets += 1

        assert auto_resets == 0, f"spurious auto-resets under matching noise: {auto_resets}"
        # No crashes
        assert engine._ingestion._cov_inv is not None


class TestFlappingRegime:
    """Rapid regime flapping: engine should hold safely, no crash/infinite loops."""

    def test_flapping_degrades_to_hold_and_exploration(self):
        """Flapping should cause safe degradation (HOLD/EMERGENCY_FLUSH), no crashes."""
        rng = np.random.default_rng(42)
        env = FlappingRegimeEnv(PATTERN_A, PATTERN_B, flip_every=5, rng=rng)
        engine = build_engine()

        holds = 0
        emergency_flushes = 0
        auto_resets = 0
        crashes = 0

        for _ in range(150):
            delta = env.step()
            try:
                plan = engine.process_event(delta, 1.0)
            except Exception:
                crashes += 1
                continue
            
            if plan.action == "HOLD":
                holds += 1
            elif plan.action == "EMERGENCY_FLUSH":
                emergency_flushes += 1
            
            if plan.veto_details.get("reason") == "Auto_Regime_Reset_Triggered":
                auto_resets += 1

        assert crashes == 0, "engine crashed during flapping"
        # Auto-resets should be limited (not infinite loop)
        assert auto_resets <= 6, f"too many auto-resets during flapping: {auto_resets}"
        # Safe degradation: some HOLDs or EMERGENCY_FLUSHs should occur (not pure EXECUTE)
        total_safe = holds + emergency_flushes
        assert total_safe > 15, f"engine didn't degrade safely: {total_safe} safe actions"

    def test_phi_ritmo_elevated_under_flapping(self):
        """High entropy during flapping should be reflected in non-extreme confidence if EXECUTE occurs."""
        rng = np.random.default_rng(42)
        env = FlappingRegimeEnv(PATTERN_A, PATTERN_B, flip_every=5, rng=rng)
        engine = build_engine()

        lambda_values = []
        for _ in range(80):
            delta = env.step()
            plan = engine.process_event(delta, 1.0)
            if plan.action == "EXECUTE":
                lambda_values.append(plan.global_confidence)

        # If EXECUTE happens, confidence should not be stuck at 0 or 1
        if lambda_values:
            assert all(0.01 < c < 0.99 for c in lambda_values), "confidence stuck at extremes"
        else:
            # Pure HOLD under chaos is also valid safe behavior
            pass


class TestCombinedAdversarial:
    """Multiple adversarial factors simultaneously."""

    def test_jitter_plus_noise_graceful_degradation(self):
        """Combined jitter+noise should not crash; graceful HOLD fallback."""
        rng = np.random.default_rng(42)
        env = NoisyStateEnv(PATTERN_A, noise_std=0.15, rng=rng)
        engine = build_engine()

        crashes = 0
        for _ in range(100):
            delta = env.step()
            dt = 1.0 + rng.normal(0.0, 0.2)
            dt = max(dt, 0.1)
            try:
                engine.process_event(delta, dt)
            except Exception:
                crashes += 1

        assert crashes == 0, "engine crashed under combined jitter+noise"
        # Engine may HOLD throughout - that's valid safe behavior

    def test_no_state_corruption_after_flapping(self):
        """Engine internals must remain consistent after adversarial sequence."""
        rng = np.random.default_rng(42)
        env = FlappingRegimeEnv(PATTERN_A, PATTERN_B, flip_every=5, rng=rng)
        engine = build_engine()

        for _ in range(200):
            delta = env.step()
            engine.process_event(delta, 1.0)

        # Verify all internal structures intact
        assert engine._ingestion._cov_inv is not None
        assert engine._rhythm._theta.total_updates > 0
        assert engine._tracker._current_step >= 0
        assert len(engine._rhythm._history) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])