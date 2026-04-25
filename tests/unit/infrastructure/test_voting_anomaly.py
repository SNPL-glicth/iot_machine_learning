"""Tests para VotingAnomalyDetector.

Escenarios:
- Valor normal → no anomalía
- Valor extremo → anomalía con alto score
- Entrenamiento con datos insuficientes
- Votos individuales verificados
- Sensor window vacía
"""

from __future__ import annotations

import random

import pytest

from iot_machine_learning.domain.entities.anomaly import AnomalySeverity
from iot_machine_learning.domain.entities.sensor_reading import (
    SensorReading,
    SensorWindow,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.detector import (
    VotingAnomalyDetector,
)


def _make_window(series_id: str, values: list[float]) -> SensorWindow:
    """Helper para crear SensorWindow desde lista de valores."""
    readings = [
        SensorReading(series_id=series_id, value=v, timestamp=float(i))
        for i, v in enumerate(values)
    ]
    return SensorWindow(series_id=series_id, readings=readings)


class TestVotingDetection:
    """Tests de detección con voting."""

    @pytest.fixture(autouse=True)
    def _train_detector(self) -> None:
        """Entrena detector con datos normales."""
        random.seed(42)
        self.detector = VotingAnomalyDetector(
            contamination=0.1,
            voting_threshold=0.5,
        )
        # Datos normales: 20°C ± 1°C
        historical = [20.0 + random.gauss(0, 1.0) for _ in range(200)]
        self.detector.train(historical)

    def test_normal_value_not_anomaly(self) -> None:
        """Valor dentro del rango normal → no anomalía."""
        window = _make_window("1", [20.0, 20.1, 19.9, 20.2, 20.5])
        result = self.detector.detect(window)

        assert result.is_anomaly is False
        assert result.score < 0.5
        assert result.severity in (AnomalySeverity.NONE, AnomalySeverity.LOW)

    def test_extreme_value_is_anomaly(self) -> None:
        """Valor muy fuera del rango → anomalía.

        Usa pesos que garantizan detección solo con z_score + iqr,
        sin depender de sklearn IF/LOF (que pueden no estar disponibles
        o estar mockeados en la sesión de tests).
        """
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

        window = _make_window("1", [20.0, 20.1, 19.9, 20.2, 50.0])
        result = detector.detect(window)

        assert result.is_anomaly is True
        assert result.score > 0.5
        assert "z_score" in result.method_votes
        assert result.explanation != ""

    def test_method_votes_present(self) -> None:
        """Todos los métodos deben votar."""
        window = _make_window("1", [20.0, 20.1, 19.9, 20.2, 35.0])
        result = self.detector.detect(window)

        assert "z_score" in result.method_votes
        assert "iqr" in result.method_votes
        # IF y LOF dependen de sklearn, verificar si están
        # (pueden no estar en todos los entornos)

    def test_empty_window_returns_normal(self) -> None:
        """Ventana vacía → resultado normal."""
        window = SensorWindow(series_id="1", readings=[])
        result = self.detector.detect(window)

        assert result.is_anomaly is False
        assert result.score == 0.0


class TestVotingTraining:
    """Tests de entrenamiento."""

    def test_insufficient_data_raises(self) -> None:
        """Menos de 50 puntos debe fallar."""
        detector = VotingAnomalyDetector()
        with pytest.raises(ValueError, match="50"):
            detector.train([1.0] * 30)

    def test_detect_without_train_autofires_train(self) -> None:
        """Auto-entrena en primera llamada si hay suficientes datos."""
        detector = VotingAnomalyDetector()
        random.seed(42)
        values = [20.0 + random.gauss(0, 1.0) for _ in range(60)]
        window = _make_window("1", values)
        result = detector.detect(window)
        assert result is not None
        assert detector.is_trained() is True

    def test_detect_insufficient_data_returns_neutral(self) -> None:
        """Ventana pequeña sin entrenar previo → resultado neutral."""
        detector = VotingAnomalyDetector()
        window = _make_window("1", [20.0] * 10)
        result = detector.detect(window)
        assert result.is_anomaly is False
        assert result.score == 0.0
        assert "cold_start" in result.method_votes
        assert result.context["reason"] == "auto_train_skipped"

    def test_explicit_train_still_works(self) -> None:
        """Comportamiento existente sin cambios cuando train() se llama primero."""
        detector = VotingAnomalyDetector()
        random.seed(42)
        detector.train([20.0 + random.gauss(0, 1.0) for _ in range(100)])
        assert detector.is_trained() is True

        window = _make_window("1", [20.0, 20.1, 19.9, 20.2, 20.5])
        result = detector.detect(window)
        assert result.is_anomaly is False

    def test_is_trained_flag(self) -> None:
        """Flag de entrenamiento debe actualizarse."""
        detector = VotingAnomalyDetector()
        assert detector.is_trained() is False

        random.seed(42)
        detector.train([20.0 + random.gauss(0, 1) for _ in range(100)])
        assert detector.is_trained() is True


class TestVotingConstructor:
    """Validaciones del constructor."""

    def test_invalid_contamination_raises(self) -> None:
        with pytest.raises(ValueError, match="contamination"):
            VotingAnomalyDetector(contamination=0.0)

        with pytest.raises(ValueError, match="contamination"):
            VotingAnomalyDetector(contamination=0.6)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="voting_threshold"):
            VotingAnomalyDetector(voting_threshold=0.0)

        with pytest.raises(ValueError, match="voting_threshold"):
            VotingAnomalyDetector(voting_threshold=1.0)
