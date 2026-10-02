"""Tests unitarios de contratos e infraestructura de Fase 2.5 (Kuramoto Consensus Gate)."""

from __future__ import annotations

import time
from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest

from iot_machine_learning.domain.entities.consensus import (
    ConsensusDecision,
    KuramotoGateConfig,
    KuramotoState,
)
from iot_machine_learning.domain.entities.representation_evidence import (
    EvidenceScore,
    RepresentationLevel,
    SystemOperationalState,
)
from iot_machine_learning.domain.ports.meta_gate_port import (
    AdaptiveMetaGatePort,
    ConsensusGatePort,
)
from iot_machine_learning.infrastructure.ml.moe.adaptive import KuramotoConsensusGate


class TestConsensusDomainPurity:
    """Verifica la pureza absoluta de domain/entities/consensus.py y domain/ports/meta_gate_port.py."""

    def test_consensus_entities_purity(self) -> None:
        path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "domain"
            / "entities"
            / "consensus.py"
        )
        content = path.read_text(encoding="utf-8")
        forbidden = ["numpy", "scipy", "sklearn", "torch", "pandas", "infrastructure"]
        for pkg in forbidden:
            assert f"import {pkg}" not in content
            assert f"from {pkg}" not in content

    def test_meta_gate_port_purity(self) -> None:
        path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "domain"
            / "ports"
            / "meta_gate_port.py"
        )
        content = path.read_text(encoding="utf-8")
        forbidden = ["numpy", "scipy", "sklearn", "torch", "pandas", "infrastructure"]
        for pkg in forbidden:
            assert f"import {pkg}" not in content
            assert f"from {pkg}" not in content


class TestConsensusEntitiesImmutability:
    """Verifica la inmutabilidad de KuramotoGateConfig, KuramotoState y ConsensusDecision."""

    def test_config_is_frozen(self) -> None:
        cfg = KuramotoGateConfig(delta_t=0.05)
        with pytest.raises(FrozenInstanceError):
            cfg.delta_t = 0.10  # type: ignore[misc]

    def test_state_is_frozen(self) -> None:
        st = KuramotoState(
            step=1,
            order_parameter=0.5,
            global_phase=1.0,
            phase_velocity=0.0,
            phases={"exp1": 0.0},
        )
        with pytest.raises(FrozenInstanceError):
            st.order_parameter = 0.9  # type: ignore[misc]

    def test_decision_is_frozen(self) -> None:
        decision = ConsensusDecision(
            step=1,
            operational_state=SystemOperationalState.RESTING,
            order_parameter=0.2,
            phase_velocity=0.0,
            dynamic_threshold=0.85,
            is_triggered=False,
            reason="nominal",
        )
        with pytest.raises(FrozenInstanceError):
            decision.is_triggered = True  # type: ignore[misc]

        # Verificar compatibilidad duck-typing con CausalAggregator
        assert decision.martingale_value == 20.0


class TestKuramotoConsensusGate:
    """Valida la dinámica de sincronización de Adler-Kuramoto y el control NAB."""

    def test_protocol_compliance(self) -> None:
        gate = KuramotoConsensusGate(expert_names=["exp1", "exp2", "exp3"])
        assert isinstance(gate, ConsensusGatePort)
        assert isinstance(gate, AdaptiveMetaGatePort)

    def test_noise_immunity_resting(self) -> None:
        # En reposo con un solo experto ruidoso al 85%, el sistema no se sincroniza
        gate = KuramotoConsensusGate(
            expert_names=["fast_raw", "mid_2x", "slow_10x"],
            expert_levels={
                "fast_raw": RepresentationLevel.RAW,
                "mid_2x": RepresentationLevel.TWO_X,
                "slow_10x": RepresentationLevel.TEN_X,
            },
        )

        for step in range(1, 20):
            evidences = [
                EvidenceScore("fast_raw", RepresentationLevel.RAW, 0.85, 1.0),
                EvidenceScore("mid_2x", RepresentationLevel.TWO_X, 0.02, 0.2),
                EvidenceScore("slow_10x", RepresentationLevel.TEN_X, 0.01, 0.05),
            ]
            dec = gate.evaluate_step(step, evidences, SystemOperationalState.RESTING)
            assert dec.is_triggered is False
            assert dec.order_parameter < dec.dynamic_threshold

    def test_coherent_anomaly_trigger(self) -> None:
        # Ante falla donde 2 o más expertos confirman peligro, los osciladores colapsan en fase
        gate = KuramotoConsensusGate(
            expert_names=["fast_raw", "mid_2x", "slow_10x"],
        )

        triggered = False
        for step in range(1, 15):
            evidences = [
                EvidenceScore("fast_raw", RepresentationLevel.RAW, 0.95, 1.0),
                EvidenceScore("mid_2x", RepresentationLevel.TWO_X, 0.90, 0.2),
                EvidenceScore("slow_10x", RepresentationLevel.TEN_X, 0.85, 0.05),
            ]
            dec = gate.evaluate_step(step, evidences, SystemOperationalState.SHOCKED)
            if dec.is_triggered:
                triggered = True
                assert dec.order_parameter >= 0.40
                break

        assert triggered is True

    def test_topological_quenching(self) -> None:
        # Tras un disparo, se activa el quenching refractario que desincroniza el sistema
        gate = KuramotoConsensusGate(
            expert_names=["fast_raw", "mid_2x", "slow_10x"],
            config=KuramotoGateConfig(refractory_steps=3),
        )

        ev_anom = [
            EvidenceScore("fast_raw", RepresentationLevel.RAW, 0.99, 1.0),
            EvidenceScore("mid_2x", RepresentationLevel.TWO_X, 0.99, 0.2),
            EvidenceScore("slow_10x", RepresentationLevel.TEN_X, 0.99, 0.05),
        ]

        # Forzar trigger
        step_trig = None
        for step in range(1, 25):
            dec = gate.evaluate_step(step, ev_anom, SystemOperationalState.SHOCKED)
            if dec.is_triggered:
                step_trig = step
                break

        assert step_trig is not None
        # En el paso inmediato siguiente, debe entrar en quenching
        dec_next = gate.evaluate_step(step_trig + 1, ev_anom, SystemOperationalState.SHOCKED)
        assert dec_next.is_triggered is False
        assert dec_next.reason == "refractory_quenching_cooldown"

    def test_calibrate_from_warmup(self) -> None:
        gate = KuramotoConsensusGate(expert_names=["e1", "e2", "e3"])
        nominal_r = [0.05, 0.08, 0.12, 0.06, 0.10, 0.14]
        gate.calibrate_from_warmup(nominal_r)
        # Umbral RESTING debe ser mayor que el máximo observado
        assert gate.state_thresholds[SystemOperationalState.RESTING] > max(nominal_r)

    def test_submillisecond_latency(self) -> None:
        gate = KuramotoConsensusGate(expert_names=["e1", "e2", "e3"])
        ev = [EvidenceScore("e1", RepresentationLevel.RAW, 0.1, 1.0)]

        t0 = time.perf_counter()
        for step in range(500):
            gate.evaluate_step(step, ev, SystemOperationalState.RESTING)
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        latency_per_step_us = (total_time_ms / 500) * 1000.0
        assert latency_per_step_us < 97.0  # Garantía estricta < 0.097 ms (97 microsegundos)
