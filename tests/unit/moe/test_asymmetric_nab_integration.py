"""Tests unitarios de integración para el despachador MoE asimétrico y Ville Gate en NAB.

Verifica:
1. Cumplimiento estricto del contrato AsymmetricExpertPort por SubDetectorExpertAdapter.
2. Despacho condicional y registro de expertos en AsymmetricDispatcher.
3. Equivalencia funcional punto por punto entre el pipeline simétrico y el asimétrico.
4. Conexión de LatencyBudgetAwareGate (cota de Ville) como mecanismo de decisión de riesgo.
"""

from __future__ import annotations

import numpy as np
import pytest

from iot_machine_learning.domain.entities.iot.sensor_reading import Reading, SensorWindow
from iot_machine_learning.domain.entities.representation_evidence import (
    EvidenceScore,
    RepresentationLevel,
    SystemOperationalState,
)
from iot_machine_learning.domain.ports.asymmetric_expert_port import (
    AsymmetricExpertPort,
)
from iot_machine_learning.infrastructure.ml.anomaly.calibration.layer import (
    AdaptiveDetectorCalibrationLayer,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.config import (
    AnomalyDetectorConfig,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.detector import (
    VotingAnomalyDetector,
)
from iot_machine_learning.infrastructure.ml.anomaly.detectors.iqr_detector import (
    IQRDetector,
)
from iot_machine_learning.infrastructure.ml.anomaly.detectors.z_score_detector import (
    ZScoreDetector,
)
from iot_machine_learning.infrastructure.ml.moe.adaptive import (
    LatencyBudgetAwareGate,
    OnlineConformalCalibrator,
)
from iot_machine_learning.infrastructure.ml.moe.asymmetric import (
    AsymmetricDispatcher,
    SubDetectorExpertAdapter,
)


class TestSubDetectorExpertAdapter:
    """Valida el adaptador entre SubDetector y AsymmetricExpertPort."""

    def test_adapter_implements_asymmetric_expert_port(self) -> None:
        sub = ZScoreDetector()
        adapter = SubDetectorExpertAdapter(
            sub,
            affinity=RepresentationLevel.TEN_X,
            compute_cost_estimate=0.02,
        )
        assert isinstance(adapter, AsymmetricExpertPort)
        assert adapter.name == "z_score"
        assert adapter.affinity == RepresentationLevel.TEN_X
        assert adapter.compute_cost_estimate == 0.02

    def test_adapter_evaluation_returns_evidence_score(self) -> None:
        sub = IQRDetector()
        sub.train([10.0, 10.2, 9.8, 10.1, 10.0, 9.9, 10.1] * 10)
        adapter = SubDetectorExpertAdapter(
            sub,
            affinity=RepresentationLevel.TEN_X,
            compute_cost_estimate=0.01,
        )
        # Valor normal
        score_norm = adapter.evaluate([10.0])
        assert isinstance(score_norm, EvidenceScore)
        assert score_norm.expert_name == "iqr"
        assert score_norm.representation_affinity == RepresentationLevel.TEN_X
        assert score_norm.anomaly_probability == 0.0

        # Valor extremo
        score_ano = adapter.evaluate([100.0])
        assert isinstance(score_ano, EvidenceScore)
        assert score_ano.anomaly_probability > 0.5


class TestAsymmetricVotingEquivalence:
    """Valida la equivalencia matemática y funcional del pipeline asimétrico."""

    @pytest.fixture
    def synthetic_stream(self) -> tuple[list[float], list[float]]:
        np.random.seed(42)
        n = 500
        normal = list(np.random.normal(50.0, 1.0, n))
        # Inyectar una anomalía puntual y una deriva
        normal[300] = 95.0
        normal[301] = 98.0
        for k in range(400, 450):
            normal[k] += (k - 400) * 0.4
        ts = [float(1600000000 + i * 60) for i in range(n)]
        return normal, ts

    def test_asymmetric_dispatch_exact_equivalence(
        self, synthetic_stream: tuple[list[float], list[float]]
    ) -> None:
        values, timestamps = synthetic_stream
        cfg = AnomalyDetectorConfig(voting_threshold=0.65, contamination=0.01)

        # 1. Pipeline Simétrico Tradicional (evalúa todo en cada paso)
        det_sym = VotingAnomalyDetector(
            config=cfg,
            series_id="test_series",
            calibration_layer=AdaptiveDetectorCalibrationLayer(),
            enable_asymmetric_dispatch=False,
        )
        det_sym.train(values[:100], timestamps=timestamps[:100])

        # 2. Pipeline Asimétrico MoE (enrutamiento condicional)
        det_asym = VotingAnomalyDetector(
            config=cfg,
            series_id="test_series",
            calibration_layer=AdaptiveDetectorCalibrationLayer(),
            enable_asymmetric_dispatch=True,
        )
        det_asym.train(values[:100], timestamps=timestamps[:100])

        preds_sym: list[int] = []
        preds_asym: list[int] = []

        window_size = 30
        for i in range(100, len(values)):
            sl_v = values[i - window_size + 1 : i + 1]
            sl_t = timestamps[i - window_size + 1 : i + 1]
            readings = [
                Reading(series_id="test_series", value=v, timestamp=t)
                for v, t in zip(sl_v, sl_t)
            ]
            w = SensorWindow(series_id="test_series", readings=readings)

            res_sym = det_sym.detect(w)
            res_asym = det_asym.detect(w)

            preds_sym.append(int(res_sym.is_anomaly))
            preds_asym.append(int(res_asym.is_anomaly))

        # Cero diferencias en las decisiones de anomalía
        diffs = [
            i for i, (s, a) in enumerate(zip(preds_sym, preds_asym)) if s != a
        ]
        assert len(diffs) == 0, f"Discrepancia en {len(diffs)} puntos: {diffs}"

    def test_ville_gate_integration(self) -> None:
        calibrator = OnlineConformalCalibrator(
            nominal_prior_rate=0.05,
            betting_fraction=5.0,
            learning_rate=0.15,
            known_experts=["z_score", "iqr", "isolation_forest"],
        )
        gate = LatencyBudgetAwareGate(
            calibrator=calibrator,
            alpha_target=0.05,
            budget_penalty_weight=1.5,
        )

        cfg = AnomalyDetectorConfig(voting_threshold=0.65)
        detector = VotingAnomalyDetector(
            config=cfg,
            series_id="test_series",
            enable_asymmetric_dispatch=True,
            meta_gate=gate,
        )
        assert detector._meta_gate is not None
        assert detector._enable_asymmetric_dispatch is True
