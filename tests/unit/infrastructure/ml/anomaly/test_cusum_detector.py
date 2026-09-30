"""Unit tests for CumulativeResidualDetector (CUSUM sub-detector)."""
from __future__ import annotations

import random
import pytest

from iot_machine_learning.infrastructure.ml.anomaly.detectors.cusum_detector import CumulativeResidualDetector
from iot_machine_learning.infrastructure.ml.anomaly.core.detector import VotingAnomalyDetector
from iot_machine_learning.infrastructure.ml.anomaly.core.config import AnomalyDetectorConfig
from iot_machine_learning.domain.entities.iot.sensor_reading import Reading, SensorWindow


class TestCumulativeResidualDetector:

    def test_handles_insufficient_data(self) -> None:
        det = CumulativeResidualDetector()
        assert det.vote(10.0) is None
        assert det.raw_score(10.0) is None
        assert not det.is_trained

    def test_nominal_values_emit_zero_vote(self) -> None:
        rng = random.Random(42)
        nominal = [20.0 + rng.gauss(0, 0.5) for _ in range(100)]
        det = CumulativeResidualDetector(k_factor=1.0, h_factor=2.0)
        det.train(nominal)
        assert det.is_trained

        # Feed nominal observations
        votes = []
        for _ in range(25):
            val = 20.0 + rng.gauss(0, 0.5)
            votes.append(det.vote(val))

        assert all(v == 0.0 for v in votes)

    def test_persistent_plateau_detects_and_saturates(self) -> None:
        """Una meseta persistente por encima de la envolvente satura el acumulador en 1.0."""
        rng = random.Random(42)
        nominal = [20.0 + rng.gauss(0, 0.2) for _ in range(100)]
        det = CumulativeResidualDetector(k_factor=1.0, h_factor=2.0)
        det.train(nominal)

        # Inyectar meseta anómala en 28.0 (muy por encima de max nominal ~20.6)
        votes = []
        raw_scores = []
        for _ in range(20):
            raw_s = det.raw_score(28.0)
            v = det.vote(28.0)
            raw_scores.append(raw_s)
            votes.append(v)

        # Debe acumular evidencia y saturar a 1.0
        assert votes[-1] == 1.0
        assert raw_scores[-1] >= 1.0
        # Mantiene la alerta durante la segunda mitad de la meseta
        assert all(v == 1.0 for v in votes[10:])

    def test_recovers_after_nominal_resumption(self) -> None:
        """Al regresar la señal a valores nominales, el acumulador decae hacia 0.0."""
        rng = random.Random(42)
        nominal = [20.0 + rng.gauss(0, 0.2) for _ in range(80)]
        det = CumulativeResidualDetector(k_factor=1.0, h_factor=2.0, leak_rate=0.20)
        det.train(nominal)

        # Disparar alerta con meseta
        for _ in range(10):
            det.vote(28.0)
        assert det.vote(28.0) == 1.0

        # Regresar a nominales
        for _ in range(20):
            det.vote(20.0)

        # El voto debe haber retornado a 0.0
        final_vote = det.vote(20.0)
        assert final_vote == 0.0

    def test_anti_windup_bounds_accumulation(self) -> None:
        """Anti-windup asegura que S_pos no crezca indefinidamente."""
        rng = random.Random(42)
        nominal = [20.0 + rng.gauss(0, 0.2) for _ in range(60)]
        det = CumulativeResidualDetector(anti_windup_factor=2.5)
        det.train(nominal)

        for _ in range(100):
            det.vote(100.0)

        assert det._s_pos <= 2.5 * det._h + 1e-6

    def test_integration_in_voting_anomaly_detector(self) -> None:
        """Prueba que el detector CUSUM está plenamente integrado en el ensamble."""
        rng = random.Random(42)
        nominal = [20.0 + rng.gauss(0, 0.5) for _ in range(100)]
        timestamps = [float(i) for i in range(100)]

        cfg = AnomalyDetectorConfig(min_training_points=50)
        detector = VotingAnomalyDetector(config=cfg)
        detector.train(nominal, timestamps=timestamps)

        # Generar ventana anómala
        readings = [Reading(series_id="test", value=20.0, timestamp=float(i)) for i in range(49)]
        readings.append(Reading(series_id="test", value=50.0, timestamp=49.0))
        window = SensorWindow(series_id="test", readings=readings)

        res = detector.detect(window)
        assert "cumulative_residual" in res.method_votes
