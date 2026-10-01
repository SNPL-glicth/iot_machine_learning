"""Tests unitarios de topología causal multivariada, estimación online y supresión de cascadas (Fase 3)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import time
import numpy as np
import pytest

from iot_machine_learning.domain.entities.conformal_risk import (
    AdaptiveGateDecision,
    RiskCertificationStatus,
)
from iot_machine_learning.domain.entities.representation_evidence import SystemOperationalState
from iot_machine_learning.domain.entities.topology import (
    CausalEdge,
    CausalRelationType,
    RootCauseDiagnosis,
    SystemWideAlarm,
)
from iot_machine_learning.domain.ports.topology_port import (
    CausalGatingOrchestratorPort,
    CausalTopologyPort,
)
from iot_machine_learning.infrastructure.ml.topology import (
    CausalGatingAggregator,
    SparseCausalGraph,
    StreamingTransferEntropyEstimator,
)


class TestTopologyDomainPurity:
    """Verifica la pureza absoluta de domain/entities/topology.py y domain/ports/topology_port.py."""

    def test_topology_entities_purity(self) -> None:
        path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "domain"
            / "entities"
            / "topology.py"
        )
        content = path.read_text(encoding="utf-8")
        forbidden = ["numpy", "scipy", "sklearn", "torch", "pandas", "infrastructure"]
        for pkg in forbidden:
            assert f"import {pkg}" not in content
            assert f"from {pkg}" not in content

    def test_topology_port_purity(self) -> None:
        path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "domain"
            / "ports"
            / "topology_port.py"
        )
        content = path.read_text(encoding="utf-8")
        forbidden = ["numpy", "scipy", "sklearn", "torch", "pandas", "infrastructure"]
        for pkg in forbidden:
            assert f"import {pkg}" not in content
            assert f"from {pkg}" not in content


class TestTopologyEntitiesImmutability:
    """Verifica la inmutabilidad de CausalEdge, RootCauseDiagnosis y SystemWideAlarm."""

    def test_causal_edge_is_frozen(self) -> None:
        edge = CausalEdge(
            source_series_id="sensor_A",
            target_series_id="sensor_B",
            lag_steps=10,
            coupling_strength=0.85,
        )
        with pytest.raises(FrozenInstanceError):
            edge.lag_steps = 15  # type: ignore[misc]

    def test_root_cause_diagnosis_is_frozen(self) -> None:
        diag = RootCauseDiagnosis(
            root_series_id="sensor_A",
            dominant_mechanism="shock_raw",
            mechanism_confidence=0.80,
            detection_step=100,
            operational_state=SystemOperationalState.SHOCKED,
            active_expert_weights={"shock_raw": 0.80, "drift_2x": 0.20},
        )
        with pytest.raises(FrozenInstanceError):
            diag.dominant_mechanism = "drift_2x"  # type: ignore[misc]

    def test_system_wide_alarm_is_frozen(self) -> None:
        diag = RootCauseDiagnosis(
            root_series_id="sensor_A",
            dominant_mechanism="shock_raw",
            mechanism_confidence=0.80,
            detection_step=100,
            operational_state=SystemOperationalState.SHOCKED,
            active_expert_weights={"shock_raw": 0.80},
        )
        alarm = SystemWideAlarm(
            alarm_id="alarm_1",
            root_cause=diag,
            affected_series_ids=("sensor_A", "sensor_B"),
            suppressed_cascade_count=1,
            peak_martingale_value=250.0,
            total_compute_saved_by_suppression=1.0,
            explanation_summary="Root cause sensor_A",
        )
        with pytest.raises(FrozenInstanceError):
            alarm.suppressed_cascade_count = 2  # type: ignore[misc]


class TestSparseCausalGraph:
    """Valida la indexación dispersa y búsqueda de ancestros multi-hop con acumulación de retardo."""

    def test_protocol_compliance(self) -> None:
        graph = SparseCausalGraph()
        assert isinstance(graph, CausalTopologyPort)

    def test_multi_hop_ancestor_and_lag_accumulation(self) -> None:
        graph = SparseCausalGraph(coupling_threshold=0.3)
        # Red: A --(lag=10, s=0.9)--> B --(lag=15, s=0.8)--> C
        graph.add_edge(CausalEdge("A", "B", lag_steps=10, coupling_strength=0.9))
        graph.add_edge(CausalEdge("B", "C", lag_steps=15, coupling_strength=0.8))

        # A precede a B
        is_anc, lag, s = graph.is_ancestor("A", "B")
        assert is_anc is True
        assert lag == 10
        assert s == 0.9

        # A precede a C (multi-hop)
        is_anc, lag, s = graph.is_ancestor("A", "C")
        assert is_anc is True
        assert lag == 25  # 10 + 15
        assert s == 0.8

        # C NO precede a A (direccionalidad estricta)
        is_anc_rev, _, _ = graph.is_ancestor("C", "A")
        assert is_anc_rev is False

        # Nodo no relacionado D
        is_anc_d, _, _ = graph.is_ancestor("A", "D")
        assert is_anc_d is False


class TestStreamingTransferEntropyEstimator:
    """Valida la estimación online de retardo y direccionalidad causal."""

    def test_protocol_compliance(self) -> None:
        estimator = StreamingTransferEntropyEstimator()
        assert isinstance(estimator, CausalTopologyPort)

    def test_dynamic_coupling_identification(self) -> None:
        np.random.seed(42)
        estimator = StreamingTransferEntropyEstimator(
            max_lag=15,
            window_size=150,
            coupling_threshold=0.4,
            update_interval=5,
            min_samples=40,
        )

        # Generar señal AR(1): B es una versión retardada de A (delta = 8) más ruido leve
        n_steps = 150
        e = np.random.randn(n_steps)
        signal_a = np.zeros(n_steps)
        for t in range(1, n_steps):
            signal_a[t] = 0.85 * signal_a[t - 1] + e[t]

        lag_true = 8
        signal_b = np.zeros(n_steps)
        signal_b[lag_true:] = 0.9 * signal_a[:-lag_true] + 0.1 * np.random.randn(n_steps - lag_true)

        for t in range(n_steps):
            estimator.register_pair_observation("sensor_A", "sensor_B", signal_a[t], signal_b[t], t)

        active_edges = estimator.get_active_edges()
        assert len(active_edges) == 1
        edge = active_edges[0]
        assert edge.source_series_id == "sensor_A"
        assert edge.target_series_id == "sensor_B"
        assert abs(edge.lag_steps - lag_true) <= 1
        assert edge.coupling_strength >= 0.5


class TestCausalGatingAggregator:
    """Valida la supresión de cascadas y diagnóstico RCA bajo el orquestador."""

    def test_protocol_compliance(self) -> None:
        graph = SparseCausalGraph()
        aggregator = CausalGatingAggregator(graph)
        assert isinstance(aggregator, CausalGatingOrchestratorPort)

    def _make_dummy_decision(
        self,
        step: int,
        is_triggered: bool,
        martingale_value: float,
        weights: dict[str, float],
    ) -> AdaptiveGateDecision:
        return AdaptiveGateDecision(
            step=step,
            operational_state=SystemOperationalState.SHOCKED if is_triggered else SystemOperationalState.RESTING,
            martingale_value=martingale_value,
            dynamic_threshold=100.0,
            certification=RiskCertificationStatus.CERTIFIED_ALARM if is_triggered else RiskCertificationStatus.NOMINAL,
            is_triggered=is_triggered,
            active_expert_weights=weights,
            budget_penalty_factor=1.0,
            reason="test",
        )

    def test_cascade_suppression_and_rca(self) -> None:
        # Grafo: A --(lag=10)--> B
        graph = SparseCausalGraph()
        graph.add_edge(CausalEdge("A", "B", lag_steps=10, coupling_strength=0.9))

        aggregator = CausalGatingAggregator(
            graph,
            cooldown_steps=30,
            lag_tolerance=5,
            nominal_expert_cost=1.0,
        )

        weights_a = {"regime_shift_2x": 0.85, "high_frequency_raw": 0.15}
        weights_b = {"high_frequency_raw": 0.90, "regime_shift_2x": 0.10}

        # 1. Paso 100: A dispara alerta (raíz)
        dec_a_100 = self._make_dummy_decision(100, is_triggered=True, martingale_value=150.0, weights=weights_a)
        dec_b_100 = self._make_dummy_decision(100, is_triggered=False, martingale_value=5.0, weights=weights_b)

        alarms_100 = aggregator.process_decisions(100, {"A": dec_a_100, "B": dec_b_100})
        assert len(alarms_100) == 1
        alarm = alarms_100[0]
        assert alarm.root_cause.root_series_id == "A"
        assert alarm.root_cause.dominant_mechanism == "regime_shift_2x"
        assert alarm.root_cause.mechanism_confidence == 0.85
        assert alarm.suppressed_cascade_count == 0
        assert alarm.affected_series_ids == ("A",)

        # 2. Paso 110: B dispara alerta (efecto con retardo delta=10)
        dec_a_110 = self._make_dummy_decision(110, is_triggered=False, martingale_value=20.0, weights=weights_a)
        dec_b_110 = self._make_dummy_decision(110, is_triggered=True, martingale_value=180.0, weights=weights_b)

        alarms_110 = aggregator.process_decisions(110, {"A": dec_a_110, "B": dec_b_110})
        assert len(alarms_110) == 1
        consolidated = alarms_110[0]
        # La alerta individual de B fue suprimida y consolidada en el incidente original de A
        assert consolidated.root_cause.root_series_id == "A"
        assert consolidated.suppressed_cascade_count == 1
        assert "B" in consolidated.affected_series_ids
        assert consolidated.peak_martingale_value == 180.0
        assert aggregator.total_raw_alerts_received == 2
        assert aggregator.total_cascade_alerts_suppressed == 1
        assert aggregator.cascade_suppression_rate == 0.5

    def test_uncoupled_node_triggers_separate_alarm(self) -> None:
        # Grafo con A -> B, pero C no acoplado
        graph = SparseCausalGraph()
        graph.add_edge(CausalEdge("A", "B", lag_steps=10, coupling_strength=0.9))

        aggregator = CausalGatingAggregator(graph, cooldown_steps=20)

        # A dispara en 50
        dec_a = self._make_dummy_decision(50, is_triggered=True, martingale_value=120.0, weights={"exp1": 1.0})
        aggregator.process_decisions(50, {"A": dec_a})

        # C dispara en 52 (no es descendiente de A -> debe ser una alarma independiente)
        dec_c = self._make_dummy_decision(52, is_triggered=True, martingale_value=140.0, weights={"exp2": 1.0})
        alarms_52 = aggregator.process_decisions(52, {"C": dec_c})

        assert len(alarms_52) == 1
        assert alarms_52[0].root_cause.root_series_id == "C"
        assert alarms_52[0].suppressed_cascade_count == 0

    def test_submillisecond_latency(self) -> None:
        graph = SparseCausalGraph()
        for i in range(5):
            graph.add_edge(CausalEdge(f"N{i}", f"N{i+1}", lag_steps=5, coupling_strength=0.8))

        aggregator = CausalGatingAggregator(graph)
        decisions = {
            f"N{i}": self._make_dummy_decision(i, is_triggered=(i == 0), martingale_value=110.0, weights={"exp": 1.0})
            for i in range(6)
        }

        # Ejecutar 100 iteraciones y medir tiempo por llamada
        t0 = time.perf_counter()
        for step in range(100):
            aggregator.process_decisions(step, decisions)
        elapsed_total = time.perf_counter() - t0
        latency_per_step_ms = (elapsed_total / 100) * 1000.0

        # Debe ser estrictamente menor a 1.0 ms (típicamente < 0.05 ms)
        assert latency_per_step_ms < 1.0
