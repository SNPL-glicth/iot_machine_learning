"""Tests unitarios de contratos de dominio y cumplimiento de arquitectura para Fase 1."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest

from iot_machine_learning.domain.entities.representation_evidence import (
    EvidenceScore,
    PolicyDecision,
    RepresentationLevel,
    SystemOperationalState,
)
from iot_machine_learning.domain.ports.asymmetric_expert_port import AsymmetricExpertPort
from iot_machine_learning.domain.ports.representation_policy_port import (
    BaseRepresentationPolicyPort,
    BaseSentinelPort,
)
from iot_machine_learning.infrastructure.ml.representation import (
    AgnosticRepresentationPolicy,
    EmpiricalDistributionProfile,
    GenericRegimeShiftSentinel,
    GenericShockSentinel,
    RegretRingBuffer,
)


class TestDomainContractsPurity:
    """Verifica que el dominio no importe infraestructura ni dependencias externas."""

    def test_domain_entities_have_no_third_party_imports(self) -> None:
        file_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "domain"
            / "entities"
            / "representation_evidence.py"
        )
        content = file_path.read_text(encoding="utf-8")
        forbidden = ["numpy", "scipy", "sklearn", "torch", "pandas", "infrastructure"]
        for pkg in forbidden:
            assert f"import {pkg}" not in content
            assert f"from {pkg}" not in content

    def test_domain_ports_have_no_third_party_imports(self) -> None:
        ports_dir = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "domain"
            / "ports"
        )
        for fname in ["representation_policy_port.py", "asymmetric_expert_port.py"]:
            content = (ports_dir / fname).read_text(encoding="utf-8")
            forbidden = ["numpy", "scipy", "sklearn", "torch", "pandas", "infrastructure"]
            for pkg in forbidden:
                assert f"import {pkg}" not in content
                assert f"from {pkg}" not in content


class TestDomainEntitiesImmutability:
    """Verifica inmutabilidad estricta de las entidades del dominio."""

    def test_policy_decision_is_frozen(self) -> None:
        decision = PolicyDecision(
            level=RepresentationLevel.TEN_X,
            operational_state=SystemOperationalState.RESTING,
            reason="nominal",
        )
        assert decision.level == RepresentationLevel.TEN_X
        with pytest.raises(FrozenInstanceError):
            decision.level = RepresentationLevel.RAW  # type: ignore[misc]

    def test_evidence_score_is_frozen(self) -> None:
        score = EvidenceScore(
            expert_name="cusum_raw",
            representation_affinity=RepresentationLevel.RAW,
            anomaly_probability=0.88,
            compute_cost_estimate=1.2,
        )
        assert score.anomaly_probability == 0.88
        with pytest.raises(FrozenInstanceError):
            score.anomaly_probability = 0.50  # type: ignore[misc]


class TestInfrastructureCompliance:
    """Verifica que la infraestructura implemente los contratos de dominio."""

    def test_policy_implements_representation_policy_port(self) -> None:
        profile = EmpiricalDistributionProfile(
            sample_size=100,
            median=80.0,
            q_low=60.0,
            q_high=95.0,
            q_shock_high=2.5,
            quantiles_raw={0.10: 65.0, 0.90: 90.0},
            interquartile_range=15.0,
            support_min=50.0,
            support_max=100.0,
        )
        policy = AgnosticRepresentationPolicy(
            level_profile=profile,
            shock_profile=profile,
            block_size=5,
        )

        assert isinstance(policy, BaseRepresentationPolicyPort)

        # Evaluar un step
        dec = policy.step(point=81.0, index=0)
        assert isinstance(dec, PolicyDecision)
        level, s_slice = policy.get_effective_stream_slice()
        assert isinstance(level, RepresentationLevel)

    def test_regret_ring_buffer_operations(self) -> None:
        buf = RegretRingBuffer(capacity=10)
        for i in range(15):
            buf.append(i, float(i), float(i), f"ts_{i}")

        assert len(buf.buffer) == 10
        recent = buf.get_recent_raw_slice(5)
        assert len(recent) == 5
        assert recent[-1][0] == 14
