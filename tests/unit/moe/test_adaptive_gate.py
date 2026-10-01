"""Tests unitarios de contratos e infraestructura de Fase 2 (Adaptive Meta-Gating & Conformal Risk)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest

from iot_machine_learning.domain.entities.conformal_risk import (
    AdaptiveGateDecision,
    ConformalBound,
    RiskCertificationStatus,
)
from iot_machine_learning.domain.entities.representation_evidence import (
    EvidenceScore,
    RepresentationLevel,
    SystemOperationalState,
)
from iot_machine_learning.domain.ports.meta_gate_port import (
    AdaptiveMetaGatePort,
    OnlineCalibratorPort,
)
from iot_machine_learning.infrastructure.ml.moe.adaptive import (
    LatencyBudgetAwareGate,
    OnlineConformalCalibrator,
)


class TestConformalDomainPurity:
    """Verifica la pureza absoluta de domain/entities/conformal_risk.py y domain/ports/meta_gate_port.py."""

    def test_conformal_risk_purity(self) -> None:
        path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "domain"
            / "entities"
            / "conformal_risk.py"
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


class TestConformalEntitiesImmutability:
    """Verifica la inmutabilidad de ConformalBound y AdaptiveGateDecision."""

    def test_conformal_bound_is_frozen(self) -> None:
        bound = ConformalBound(
            alpha_target=0.01,
            base_threshold=100.0,
            current_dynamic_threshold=120.0,
        )
        with pytest.raises(FrozenInstanceError):
            bound.current_dynamic_threshold = 150.0  # type: ignore[misc]

    def test_adaptive_gate_decision_is_frozen(self) -> None:
        decision = AdaptiveGateDecision(
            step=1,
            operational_state=SystemOperationalState.RESTING,
            martingale_value=1.2,
            dynamic_threshold=100.0,
            certification=RiskCertificationStatus.NOMINAL,
            is_triggered=False,
            active_expert_weights={"exp1": 1.0},
            budget_penalty_factor=1.0,
            reason="nominal",
        )
        with pytest.raises(FrozenInstanceError):
            decision.is_triggered = True  # type: ignore[misc]


class TestOnlineConformalCalibrator:
    """Valida el cálculo de e-values y la actualización adaptativa Hedge."""

    def test_protocol_compliance(self) -> None:
        calibrator = OnlineConformalCalibrator()
        assert isinstance(calibrator, OnlineCalibratorPort)

    def test_e_values_computation(self) -> None:
        calibrator = OnlineConformalCalibrator(nominal_prior_rate=0.05, betting_fraction=5.0)

        # Evidencia nominal: p = 0.01 -> e-value < 1.0
        ev_nom = EvidenceScore("exp1", RepresentationLevel.TEN_X, 0.01, 0.05)
        # Evidencia anómala: p = 0.90 -> e-value >> 1.0
        ev_anom = EvidenceScore("exp2", RepresentationLevel.RAW, 0.90, 1.0)

        e_vals = calibrator.compute_e_values([ev_nom, ev_anom])
        assert len(e_vals) == 2
        assert e_vals[0] < 1.0  # Shrinkage bajo H0
        assert e_vals[1] > 4.0  # Crecimiento de evidencia bajo H1

    def test_hedge_weight_adaptation(self) -> None:
        calibrator = OnlineConformalCalibrator(
            known_experts=["noisy_expert", "reliable_expert"],
            learning_rate=0.3,
        )

        # Simular 5 pasos en RESTING donde noisy_expert emite falsas alarmas (p=0.85)
        # y reliable_expert emite p=0.01
        for _ in range(5):
            ev_noisy = EvidenceScore("noisy_expert", RepresentationLevel.TEN_X, 0.85, 0.05)
            ev_reliable = EvidenceScore("reliable_expert", RepresentationLevel.TEN_X, 0.01, 0.05)
            e_vals = calibrator.compute_e_values([ev_noisy, ev_reliable])
            calibrator.update_weights([ev_noisy, ev_reliable], e_vals, SystemOperationalState.RESTING)

        weights = calibrator.current_weights
        assert weights["reliable_expert"] > weights["noisy_expert"]
        assert pytest.approx(sum(weights.values()), rel=1e-5) == 1.0


class TestLatencyBudgetAwareGate:
    """Valida el proceso de martingala de Ville y el umbral adaptativo."""

    def test_protocol_compliance(self) -> None:
        calibrator = OnlineConformalCalibrator()
        gate = LatencyBudgetAwareGate(calibrator)
        assert isinstance(gate, AdaptiveMetaGatePort)

    def test_budget_and_state_threshold_modulation(self) -> None:
        calibrator = OnlineConformalCalibrator()
        gate = LatencyBudgetAwareGate(calibrator, alpha_target=0.01)  # Base tau = 100

        ev = [EvidenceScore("exp1", RepresentationLevel.TEN_X, 0.01, 0.05)]

        # 1. En RESTING y 100% budget -> tau = 100 * 1.5 = 150
        dec_resting = gate.evaluate_step(1, ev, SystemOperationalState.RESTING, budget_remaining_ratio=1.0)
        assert dec_resting.dynamic_threshold == 150.0

        # 2. En SHOCKED y 100% budget -> tau = 100 * 0.6 = 60
        dec_shock = gate.evaluate_step(2, ev, SystemOperationalState.SHOCKED, budget_remaining_ratio=1.0)
        assert dec_shock.dynamic_threshold == 60.0

        # 3. Bajo estrés de budget (budget_remaining_ratio = 0.5, penalty_weight = 1.5)
        # penalty = 1.0 + 1.5 * 0.5 = 1.75 -> tau = 100 * 1.75 * 1.0 = 175
        dec_stressed = gate.evaluate_step(3, ev, SystemOperationalState.DRIFTING, budget_remaining_ratio=0.5)
        assert dec_stressed.dynamic_threshold == 175.0

    def test_certified_alarm_trigger(self) -> None:
        calibrator = OnlineConformalCalibrator()
        gate = LatencyBudgetAwareGate(calibrator, alpha_target=0.05)  # Base tau = 20

        # Evidencia persistente muy alta
        ev = [EvidenceScore("shock_raw", RepresentationLevel.RAW, 0.99, 1.0)]

        decision = None
        for step in range(1, 10):
            decision = gate.evaluate_step(step, ev, SystemOperationalState.SHOCKED)
            if decision.is_triggered:
                break

        assert decision is not None
        assert decision.is_triggered is True
        assert decision.certification == RiskCertificationStatus.CERTIFIED_ALARM
        assert "ville_threshold_exceeded" in decision.reason
