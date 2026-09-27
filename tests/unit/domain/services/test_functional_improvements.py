"""Tests unitarios para las mejoras funcionales implementadas sin tocar Ramanujan.

Valida:
1. StreamingNormalizer: normalización Z-score adaptativa online y persistencia.
2. RiskEngineAdapter con Adaptive Conformal Inference (ACI).
3. RosaRojaEngine con compuerta de Edge Económico Neto.
4. RhythmTrajectoryGenerator con memoria persistente de atractores.
5. divergence_compass con descomposición espectral fina del Jacobiano 3D.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from domain.entities.rosa_roja.movement import Movement
from domain.entities.rosa_roja.trajectory import TerminalState, Trajectory
from domain.services.normalization.streaming_normalizer import StreamingNormalizer
from domain.services.manifold.divergence_compass import classify_detailed_spectral_regime
from infrastructure.ml.adapters.risk_adapter import RiskEngineAdapter
from infrastructure.ml.engines.rosa_roja.algorithms.engine import RosaRojaEngine
from infrastructure.ml.engines.rosa_roja.algorithms.modules.rhythm_generator import RhythmTrajectoryGenerator


class TestStreamingNormalizer:
    """Verifica la normalización dimensional adaptativa online (Welford)."""

    def test_warmup_and_convergence(self) -> None:
        norm = StreamingNormalizer(dimension=1, warmup_samples=5)
        # Durante warmup devuelve ceros
        for _ in range(4):
            z = norm.update(10.0)
            assert np.allclose(z, 0.0)

        # Generar datos normales con media 50 y std 10
        np.random.seed(42)
        samples = np.random.normal(loc=50.0, scale=10.0, size=200)
        for s in samples:
            norm.update(s)

        assert pytest.approx(50.0, abs=1.5) == norm.mean[0]
        assert pytest.approx(10.0, abs=1.5) == norm.std[0]

        # Transformación Z-score
        z_val = norm.transform(60.0)
        assert pytest.approx(1.0, abs=0.2) == z_val[0]

        # Reconstrucción inversa
        reconstructed = norm.inverse_transform(z_val)
        assert pytest.approx(60.0, abs=1e-5) == reconstructed[0]

    def test_multidimensional_and_persistence(self) -> None:
        norm = StreamingNormalizer(dimension=3, warmup_samples=3)
        data = [[10.0, 100.0, 1.0], [20.0, 200.0, 2.0], [30.0, 300.0, 3.0]]
        for row in data:
            norm.update(row)

        state = norm.export_state()
        norm2 = StreamingNormalizer(dimension=3)
        norm2.import_state(state)

        assert norm2.count == norm.count
        assert np.allclose(norm2.mean, norm.mean)
        assert np.allclose(norm2.variance, norm.variance)


class TestAdaptiveConformalInferenceRisk:
    """Valida la inferencia conformal adaptativa (ACI) en el adaptador de riesgo."""

    def test_conformal_expansion_on_breaches(self) -> None:
        adapter = RiskEngineAdapter(
            l_max=0.10,
            default_sigma=0.01,
            enable_conformal=True,
            conformal_step=0.05,
            conformal_alpha=0.05,
        )
        assert adapter._q_conformal == 1.0

        # Disparar 10 violaciones consecutivas
        for i in range(10):
            adapter.record_observation(
                return_signal=0.001 + (i + 1) * 0.05,
                delta_time=1.0,
                log_return=0.01,
                expected_return=0.001,
            )

        # q_conformal debió expandirse por encima de 1.0 para compensar colas pesadas
        verdict = adapter.record_observation(return_signal=0.001, delta_time=1.0)
        assert verdict["q_conformal"] > 1.2
        assert verdict["enable_conformal"] is True

    def test_backward_compatibility_when_disabled(self) -> None:
        adapter = RiskEngineAdapter(enable_conformal=False, default_sigma=0.01)
        v = adapter.record_observation(return_signal=0.001, delta_time=1.0)
        assert v["q_conformal"] == 1.0
        assert v["enable_conformal"] is False


class TestRosaRojaNetEdgeGate:
    """Valida que la compuerta de edge neto frene ejecuciones inviables tras fricción."""

    def test_hold_when_edge_does_not_cover_friction(self) -> None:
        mock_ingestion = MagicMock()
        mock_rhythm = MagicMock()
        mock_gating = MagicMock()

        engine = RosaRojaEngine(
            ingestion_filter=mock_ingestion,
            rhythm_generator=mock_rhythm,
            moe_gating=mock_gating,
            expert_jury=[],
            drift_sensors=[],
            min_net_edge=0.0005,      # Requiere 5 bps netos mínimos
            friction_cost=0.0015,     # 15 bps de spread + comisiones
        )

        # Movimiento proyectado de 10 bps con confianza moderada (0.55)
        # Edge bruto = (2 * 0.55 - 1) * 0.0010 = 0.0001 (1 bp)
        # Edge neto = 1 bp - 15 bps = -14 bps < 5 bps -> HOLD
        m = Movement.from_raw(np.array([0.001, 0.0]), delta_time=1.0, timestamp=1.0)
        traj = Trajectory(
            movements=(m, m),
            coherence_score=0.55,
            invalidation_step=None,
            terminal_state=TerminalState(state_vector=np.array([0.001, 0.0]), step_index=1, confidence=0.55),
        )

        action = engine._determine_action(phi_moe=0.55, trajectory=traj)
        assert action == "HOLD"

    def test_execute_when_edge_is_sufficient(self) -> None:
        engine = RosaRojaEngine(
            ingestion_filter=MagicMock(),
            rhythm_generator=MagicMock(),
            moe_gating=MagicMock(),
            expert_jury=[],
            drift_sensors=[],
            min_net_edge=0.0005,
            friction_cost=0.0005,
        )
        # Movimiento proyectado amplio de 50 bps con confianza 0.80
        # Edge bruto = (2 * 0.80 - 1) * 0.0050 = 0.0030 (30 bps)
        # Edge neto = 30 bps - 5 bps = 25 bps > 5 bps -> EXECUTE
        m = Movement.from_raw(np.array([0.005, 0.0]), delta_time=1.0, timestamp=1.0)
        traj = Trajectory(
            movements=(m, m),
            coherence_score=0.80,
            invalidation_step=None,
            terminal_state=TerminalState(state_vector=np.array([0.005, 0.0]), step_index=1, confidence=0.80),
        )

        action = engine._determine_action(phi_moe=0.80, trajectory=traj)
        assert action == "EXECUTE"


class TestPersistentAttractorGraph:
    """Valida que el generador de ritmo retenga transiciones aprendidas con persistent_graph."""

    def test_persistent_memory_retains_edges(self) -> None:
        gen = RhythmTrajectoryGenerator(persistent_graph=True, max_transitions_per_key=10)
        m1 = Movement.from_raw(np.array([1.0, 0.0]), delta_time=1.0, timestamp=1.0)
        m2 = Movement.from_raw(np.array([2.0, 0.0]), delta_time=1.0, timestamp=2.0)

        gen._history = [m1, m2]
        gen._update_transition_graph()
        k1 = gen._quantize_state(m1.delta_state)
        assert len(gen._transition_graph[k1]) == 1

        # Nuevo paso
        m3 = Movement.from_raw(np.array([3.0, 0.0]), delta_time=1.0, timestamp=3.0)
        gen._history = [m2, m3]
        gen._update_transition_graph()

        # k1 se preservó gracias a persistent_graph en vez de vaciarse
        assert k1 in gen._transition_graph


class TestSpectralJacobianDecomposition:
    """Valida la clasificación espectral del Jacobiano (oscilatorio vs sumidero vs caos)."""

    def test_oscillatory_limit_cycle_detection(self) -> None:
        # Autovalores con componente imaginaria dominante y parte real casi nula
        eigvals = np.array([-0.001 + 2.5j, -0.001 - 2.5j, -0.5 + 0.0j])
        unstable, regime, metrics = classify_detailed_spectral_regime(eigvals, divergence=-0.5)
        assert unstable is False
        assert regime == "LIMIT_CYCLE_OSCILLATORY"
        assert metrics["max_imag_eig"] == pytest.approx(2.5)

    def test_expansive_chaos_detection(self) -> None:
        eigvals = np.array([1.5 + 0.0j, -0.2 + 0.0j, -0.3 + 0.0j])
        unstable, regime, _ = classify_detailed_spectral_regime(eigvals, divergence=1.0)
        assert unstable is True
        assert regime == "EXPANSIVE_CHAOS"
