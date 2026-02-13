"""Tests para VotingAnomalyDetector — detección temporal-aware.

Cubre:
- Entrenamiento con timestamps activa features temporales
- Entrenamiento sin timestamps mantiene comportamiento legacy
- Detección de spike abrupto por velocidad (valor en rango, velocidad anómala)
- Detección de aceleración anómala
- Votos temporales presentes en method_votes
- Explicación incluye velocidad/aceleración
- Multi-dim IF/LOF temporal entrenados y votando
- Backward compatibility: train(values) sin timestamps sigue funcionando
- TemporalTrainingStats correctas
- Señal gradual (normal) vs spike abrupto (anómalo) — el caso clave
"""

from __future__ import annotations

import random

import pytest

from iot_machine_learning.domain.entities.anomaly import AnomalySeverity
from iot_machine_learning.domain.entities.sensor_reading import (
    SensorReading,
    SensorWindow,
)
from iot_machine_learning.infrastructure.ml.anomaly.statistical_methods import (
    TemporalTrainingStats,
    compute_temporal_training_stats,
)
from iot_machine_learning.infrastructure.ml.anomaly.voting_anomaly_detector import (
    VotingAnomalyDetector,
)
from iot_machine_learning.infrastructure.ml.anomaly.anomaly_narrator import (
    build_anomaly_explanation,
)


def _make_window(sensor_id: int, values: list[float], timestamps: list[float] | None = None) -> SensorWindow:
    """Helper para crear SensorWindow desde listas."""
    if timestamps is None:
        timestamps = [float(i) for i in range(len(values))]
    readings = [
        SensorReading(sensor_id=sensor_id, value=v, timestamp=t)
        for v, t in zip(values, timestamps)
    ]
    return SensorWindow(sensor_id=sensor_id, readings=readings)


def _make_normal_data(n: int = 200, seed: int = 42) -> tuple[list[float], list[float]]:
    """Genera datos normales: 20°C ± 1°C con drift lento, dt=1s."""
    random.seed(seed)
    values = []
    v = 20.0
    for _ in range(n):
        v += random.gauss(0, 0.3)  # drift lento
        values.append(v)
    timestamps = [float(i) for i in range(n)]
    return values, timestamps


# ── TemporalTrainingStats ─────────────────────────────────────────────


class TestTemporalTrainingStats:
    """Tests para compute_temporal_training_stats."""

    def test_computes_velocity_stats(self):
        values, timestamps = _make_normal_data(200)
        stats = compute_temporal_training_stats(values, timestamps)

        assert stats.has_temporal is True
        assert stats.vel_std > 0
        assert stats.vel_iqr >= 0

    def test_computes_acceleration_stats(self):
        values, timestamps = _make_normal_data(200)
        stats = compute_temporal_training_stats(values, timestamps)

        assert stats.acc_std > 0
        assert stats.acc_iqr >= 0

    def test_insufficient_data_returns_empty(self):
        stats = compute_temporal_training_stats([1.0, 2.0], [0.0, 1.0])
        assert stats.has_temporal is False

    def test_empty_factory(self):
        stats = TemporalTrainingStats.empty()
        assert stats.has_temporal is False
        assert stats.vel_mean == 0.0

    def test_linear_signal_low_velocity_std(self):
        # Señal perfectamente lineal → velocidad constante → std ≈ 0
        values = [float(i) for i in range(100)]
        timestamps = [float(i) for i in range(100)]
        stats = compute_temporal_training_stats(values, timestamps)

        assert stats.has_temporal is True
        # velocity should be ~1.0 everywhere, std near epsilon
        assert abs(stats.vel_mean - 1.0) < 0.01
        assert stats.vel_std < 0.01


# ── Backward Compatibility ─────────────────────────────────────────────


class TestBackwardCompatibility:
    """Train sin timestamps debe funcionar exactamente como antes."""

    def test_train_without_timestamps(self):
        random.seed(42)
        detector = VotingAnomalyDetector(contamination=0.1, voting_threshold=0.5)
        historical = [20.0 + random.gauss(0, 1.0) for _ in range(200)]
        detector.train(historical)

        assert detector.is_trained() is True
        assert detector._temporal_stats.has_temporal is False

    def test_detect_without_temporal_no_velocity_votes(self):
        random.seed(42)
        detector = VotingAnomalyDetector(contamination=0.1, voting_threshold=0.5)
        detector.train([20.0 + random.gauss(0, 1.0) for _ in range(200)])

        window = _make_window(1, [20.0, 20.1, 19.9, 20.2, 20.5])
        result = detector.detect(window)

        assert "velocity_z" not in result.method_votes
        assert "acceleration_z" not in result.method_votes
        assert result.is_anomaly is False

    def test_extreme_value_still_detected_without_temporal(self):
        random.seed(42)
        detector = VotingAnomalyDetector(
            contamination=0.1,
            voting_threshold=0.5,
            weights={
                "z_score": 0.50,
                "iqr": 0.30,
                "isolation_forest": 0.10,
                "local_outlier_factor": 0.10,
            },
        )
        detector.train([20.0 + random.gauss(0, 1.0) for _ in range(200)])

        window = _make_window(1, [20.0, 20.1, 19.9, 20.2, 50.0])
        result = detector.detect(window)

        assert result.is_anomaly is True
        assert result.score > 0.5


# ── Temporal-Aware Detection ───────────────────────────────────────────


class TestTemporalAwareDetection:
    """Tests para detección con features temporales activadas."""

    @pytest.fixture(autouse=True)
    def _train_temporal_detector(self) -> None:
        """Entrena detector con timestamps."""
        values, timestamps = _make_normal_data(200)
        self.detector = VotingAnomalyDetector(
            contamination=0.1,
            voting_threshold=0.5,
        )
        self.detector.train(values, timestamps=timestamps)

    def test_temporal_stats_active(self):
        assert self.detector._temporal_stats.has_temporal is True

    def test_normal_value_normal_velocity_not_anomaly(self):
        """Valor normal con velocidad normal → no anomalía."""
        # Ventana con drift gradual (velocidad baja)
        window = _make_window(
            1,
            [20.0, 20.1, 20.2, 20.3, 20.4],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        )
        result = self.detector.detect(window)
        assert result.is_anomaly is False

    def test_velocity_votes_present(self):
        """Votos temporales deben estar presentes cuando entrenado con timestamps."""
        window = _make_window(
            1,
            [20.0, 20.1, 20.2, 20.3, 20.4],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        )
        result = self.detector.detect(window)
        assert "velocity_z" in result.method_votes
        assert "acceleration_z" in result.method_votes

    def test_spike_detected_by_velocity(self):
        """Spike abrupto: valor puede estar en rango pero velocidad es anómala.

        Este es el caso clave que ANO-1/ANO-3 no podían detectar:
        el valor 25°C está dentro del rango normal (20±5), pero
        llegar a 25°C en 1 segundo desde 20°C es una velocidad
        de 5°C/s, muy por encima de la velocidad normal (~0.3°C/s).
        """
        # Gradual normal, luego spike de +5 en 1 segundo
        window = _make_window(
            1,
            [20.0, 20.1, 20.2, 20.3, 25.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        )
        result = self.detector.detect(window)

        # velocity_z should be high
        assert "velocity_z" in result.method_votes
        vel_vote = result.method_votes["velocity_z"]
        assert vel_vote > 0.5, f"velocity_z vote should be high, got {vel_vote}"

    def test_acceleration_vote_on_sudden_change(self):
        """Cambio brusco de velocidad → aceleración anómala."""
        # Constante, constante, luego salto
        window = _make_window(
            1,
            [20.0, 20.0, 20.0, 20.0, 30.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        )
        result = self.detector.detect(window)

        assert "acceleration_z" in result.method_votes
        acc_vote = result.method_votes["acceleration_z"]
        assert acc_vote > 0.0  # should detect the acceleration change

    def test_gradual_rise_low_velocity_score(self):
        """Subida gradual → velocidad baja → no anomalía temporal."""
        # Subida de 20 a 25 en 50 segundos (0.1°C/s)
        n = 50
        values = [20.0 + 0.1 * i for i in range(n)]
        timestamps = [float(i) for i in range(n)]
        window = _make_window(1, values, timestamps)
        result = self.detector.detect(window)

        vel_vote = result.method_votes.get("velocity_z", 0.0)
        assert vel_vote < 0.5, f"Gradual rise should have low velocity vote, got {vel_vote}"

    def test_single_point_window_no_temporal_votes(self):
        """Ventana de 1 punto → sin votos temporales."""
        window = _make_window(1, [20.0], [0.0])
        result = self.detector.detect(window)
        assert "velocity_z" not in result.method_votes

    def test_two_point_window_has_velocity_no_acceleration(self):
        """Ventana de 2 puntos → velocity sí, acceleration no."""
        window = _make_window(1, [20.0, 30.0], [0.0, 1.0])
        result = self.detector.detect(window)
        assert "velocity_z" in result.method_votes
        # 2 points → 1 velocity → no acceleration (need ≥3 points)
        assert "acceleration_z" not in result.method_votes


# ── Narrator Temporal ──────────────────────────────────────────────────


class TestNarratorTemporal:
    """Tests para explicaciones temporales en anomaly_narrator."""

    def test_velocity_explanation(self):
        votes = {"velocity_z": 1.0}
        text = build_anomaly_explanation(votes, vel_z_score=5.2)
        assert "Velocidad anómala" in text
        assert "5.2" in text

    def test_acceleration_explanation(self):
        votes = {"acceleration_z": 1.0}
        text = build_anomaly_explanation(votes, acc_z_score=3.8)
        assert "Aceleración anómala" in text
        assert "3.8" in text

    def test_temporal_if_explanation(self):
        votes = {"isolation_forest_temporal": 1.0}
        text = build_anomaly_explanation(votes)
        assert "IF-T" in text

    def test_temporal_lof_explanation(self):
        votes = {"lof_temporal": 1.0}
        text = build_anomaly_explanation(votes)
        assert "LOF-T" in text

    def test_combined_magnitude_and_temporal(self):
        votes = {"z_score": 1.0, "velocity_z": 1.0}
        text = build_anomaly_explanation(votes, z_score=4.0, vel_z_score=6.0)
        assert "Z-score alto" in text
        assert "Velocidad anómala" in text

    def test_no_temporal_votes_normal(self):
        votes = {"z_score": 0.0, "iqr": 0.0}
        text = build_anomaly_explanation(votes)
        assert text == "Valor normal"


# ── Statistical Methods Temporal ───────────────────────────────────────


class TestStatisticalMethodsTemporal:
    """Tests para funciones temporales en statistical_methods."""

    def test_distribution_stats_correctness(self):
        from iot_machine_learning.infrastructure.ml.anomaly.statistical_methods import (
            _compute_distribution_stats,
        )
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        mean, std, q1, q3, iqr = _compute_distribution_stats(values)
        assert abs(mean - 5.5) < 0.01
        assert std > 0
        assert q1 <= q3
        assert abs(iqr - (q3 - q1)) < 1e-9

    def test_distribution_stats_empty(self):
        from iot_machine_learning.infrastructure.ml.anomaly.statistical_methods import (
            _compute_distribution_stats,
        )
        mean, std, q1, q3, iqr = _compute_distribution_stats([])
        assert mean == 0.0
        assert q1 == 0.0

    def test_temporal_stats_with_constant_signal(self):
        """Señal constante → velocidad ≈ 0, aceleración ≈ 0."""
        values = [25.0] * 100
        timestamps = [float(i) for i in range(100)]
        stats = compute_temporal_training_stats(values, timestamps)

        assert stats.has_temporal is True
        assert abs(stats.vel_mean) < 1e-6
        assert abs(stats.acc_mean) < 1e-6
