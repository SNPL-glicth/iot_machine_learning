"""Tests unitarios del Mixture of Experts Asimétrico (Fase 1).

Verifica:
- Cumplimiento de contratos AsymmetricExpertPort.
- Despacho selectivo por afinidad de representación sin condicionales cruzados.
- Acumulación secuencial de evidencia (Anytime Martingale / CUSUM con fuga).
- Cadena de inferencia completa: Stream -> Política -> Despachador -> Acumulador.
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.entities.representation_evidence import (
    EvidenceScore,
    PolicyDecision,
    RepresentationLevel,
    SystemOperationalState,
)
from iot_machine_learning.domain.ports.asymmetric_expert_port import (
    AsymmetricExpertPort,
)
from iot_machine_learning.infrastructure.ml.moe.asymmetric import (
    AsymmetricDispatcher,
    EvidenceAccumulator,
    HighFrequencyExpert,
    IntegratedEvidence,
    RegimeShiftExpert,
    RestingInvariantExpert,
)
from iot_machine_learning.infrastructure.ml.representation import (
    AgnosticRepresentationPolicy,
    EmpiricalDistributionProfile,
)


class TestAsymmetricExpertContracts:
    """Valida que todos los expertos del catálogo cumplan el contrato del dominio."""

    def test_resting_expert_conforms_to_contract(self) -> None:
        expert = RestingInvariantExpert(lower_bound=10.0, upper_bound=50.0)
        assert isinstance(expert, AsymmetricExpertPort)
        assert expert.affinity == RepresentationLevel.TEN_X
        assert expert.name == "resting_invariants_10x"

        # Evaluación en reposo nominal
        score = expert.evaluate([20.0, 25.0, 30.0])
        assert isinstance(score, EvidenceScore)
        assert score.representation_affinity == RepresentationLevel.TEN_X
        assert score.anomaly_probability < 0.05

        # Evaluación con violación de envolvente
        anomaly_score = expert.evaluate([20.0, 85.0, 30.0])
        assert anomaly_score.anomaly_probability > 0.80

    def test_regime_expert_conforms_to_contract(self) -> None:
        expert = RegimeShiftExpert(baseline_center=30.0, baseline_spread=5.0)
        assert isinstance(expert, AsymmetricExpertPort)
        assert expert.affinity == RepresentationLevel.TWO_X
        assert expert.name == "regime_shift_2x"

        # Evaluación en centroide nominal
        score = expert.evaluate([29.0, 31.0, 30.0, 30.5])
        assert score.anomaly_probability < 0.10

        # Evaluación con deriva persistente
        drift_score = expert.evaluate([45.0, 48.0, 50.0, 52.0])
        assert drift_score.anomaly_probability > 0.90

    def test_high_frequency_expert_conforms_to_contract(self) -> None:
        expert = HighFrequencyExpert(shock_threshold=2.0)
        assert isinstance(expert, AsymmetricExpertPort)
        assert expert.affinity == RepresentationLevel.RAW
        assert expert.name == "high_frequency_raw"

        # Evaluación con baja innovación
        score = expert.evaluate([10.0, 10.2, 10.1, 10.3])
        assert score.anomaly_probability < 0.10

        # Evaluación con choque abrupto
        shock_score = expert.evaluate([10.0, 10.1, 18.5, 18.6])
        assert shock_score.anomaly_probability > 0.90


class TestAsymmetricDispatcher:
    """Valida el enrutamiento selectivo por afinidad de escala."""

    @pytest.fixture
    def setup_dispatcher(self) -> tuple[AsymmetricDispatcher, RestingInvariantExpert, RegimeShiftExpert, HighFrequencyExpert]:
        exp_10x = RestingInvariantExpert(lower_bound=10.0, upper_bound=50.0)
        exp_2x = RegimeShiftExpert(baseline_center=30.0, baseline_spread=5.0)
        exp_raw = HighFrequencyExpert(shock_threshold=2.0)

        dispatcher = AsymmetricDispatcher([exp_10x, exp_2x, exp_raw])
        return dispatcher, exp_10x, exp_2x, exp_raw

    def test_selective_dispatch_by_level(self, setup_dispatcher) -> None:
        dispatcher, exp_10x, exp_2x, exp_raw = setup_dispatcher

        # 1. Despacho a 10X -> Solo ejecuta RestingInvariantExpert
        scores_10x = dispatcher.dispatch(RepresentationLevel.TEN_X, [25.0, 26.0])
        assert len(scores_10x) == 1
        assert scores_10x[0].expert_name == exp_10x.name
        assert scores_10x[0].representation_affinity == RepresentationLevel.TEN_X

        # 2. Despacho a 2X -> Solo ejecuta RegimeShiftExpert
        scores_2x = dispatcher.dispatch(RepresentationLevel.TWO_X, [28.0, 30.0, 31.0])
        assert len(scores_2x) == 1
        assert scores_2x[0].expert_name == exp_2x.name
        assert scores_2x[0].representation_affinity == RepresentationLevel.TWO_X

        # 3. Despacho a RAW -> Solo ejecuta HighFrequencyExpert
        scores_raw = dispatcher.dispatch(RepresentationLevel.RAW, [29.0, 29.5, 30.0])
        assert len(scores_raw) == 1
        assert scores_raw[0].expert_name == exp_raw.name
        assert scores_raw[0].representation_affinity == RepresentationLevel.RAW

    def test_compute_savings_telemetry(self, setup_dispatcher) -> None:
        dispatcher, exp_10x, exp_2x, exp_raw = setup_dispatcher

        # Ejecutamos 10 pasos en 10X (costo 0.05 vs hipotético 1.25 por paso)
        for _ in range(10):
            dispatcher.dispatch(RepresentationLevel.TEN_X, [25.0])

        telemetry = dispatcher.get_telemetry()
        assert telemetry["dispatches_by_level"]["10X"] == 10
        assert telemetry["dispatches_by_level"]["2X"] == 0
        assert telemetry["dispatches_by_level"]["RAW"] == 0

        # El ahorro debe ser superior al 90% operando en 10X
        assert telemetry["compute_savings_ratio"] > 0.90


class TestEvidenceAccumulator:
    """Valida la acumulación secuencial e Anytime Martingale."""

    def test_nominal_leakage(self) -> None:
        acc = EvidenceAccumulator(leak_rate=0.10)
        # Puntuaciones nominales (prob = 0.01)
        nominal_score = EvidenceScore(
            expert_name="resting_10x",
            representation_affinity=RepresentationLevel.TEN_X,
            anomaly_probability=0.01,
            compute_cost_estimate=0.05,
        )

        for i in range(10):
            verdict = acc.accumulate([nominal_score], RepresentationLevel.TEN_X, step_index=i)

        assert verdict.accumulated_evidence == 0.0
        assert verdict.martingale_value == 1.0
        assert not verdict.is_anomaly
        assert verdict.alarm_level == "NOMINAL"

    def test_sustained_anomaly_alarm(self) -> None:
        acc = EvidenceAccumulator(
            prior_nominal_prob=0.05,
            leak_rate=0.05,
            warn_threshold=2.0,
            alarm_threshold=4.0,
        )
        anomaly_score = EvidenceScore(
            expert_name="regime_2x",
            representation_affinity=RepresentationLevel.TWO_X,
            anomaly_probability=0.92,
            compute_cost_estimate=0.20,
        )

        verdict: IntegratedEvidence | None = None
        for i in range(5):
            verdict = acc.accumulate([anomaly_score], RepresentationLevel.TWO_X, step_index=i)

        assert verdict is not None
        assert verdict.accumulated_evidence >= 4.0
        assert verdict.is_anomaly is True
        assert verdict.alarm_level == "CRITICAL"
        assert verdict.dominant_expert == "regime_2x"


class TestFullInferenceChain:
    """Valida la cadena integral: Stream -> Política -> Despachador -> Acumulador."""

    def test_end_to_end_chain(self) -> None:
        # Configurar perfil empírico
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

        dispatcher = AsymmetricDispatcher([
            RestingInvariantExpert(lower_bound=50.0, upper_bound=100.0),
            RegimeShiftExpert(baseline_center=80.0, baseline_spread=15.0),
            HighFrequencyExpert(shock_threshold=5.0),
        ])

        accumulator = EvidenceAccumulator()

        # Stream nominal de 15 puntos (3 bloques de 5)
        nominal_stream = [80.0 + (i % 2) for i in range(15)]
        final_verdict = None

        for idx, pt in enumerate(nominal_stream):
            decision = policy.step(pt, idx)
            # Cuando se completa un bloque, obtenemos el slice y despachamos
            if (idx + 1) % 5 == 0:
                level, s_slice = policy.get_effective_stream_slice()
                scores = dispatcher.dispatch(level, s_slice)
                final_verdict = accumulator.accumulate(scores, level, step_index=idx // 5)

        assert final_verdict is not None
        assert not final_verdict.is_anomaly
        assert final_verdict.alarm_level == "NOMINAL"
        assert dispatcher.get_compute_savings_ratio() > 0.80
