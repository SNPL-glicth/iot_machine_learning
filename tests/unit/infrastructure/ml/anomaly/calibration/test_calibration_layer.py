"""Tests unitarios para la Capa de Calibración Adaptativa de Anomalías V1."""
import numpy as np
import pytest
from iot_machine_learning.domain.entities.iot.sensor_reading import SensorWindow

from infrastructure.ml.anomaly.calibration.guardrail import CalibrationGuardrail, CalibrationState
from infrastructure.ml.anomaly.calibration.layer import AdaptiveDetectorCalibrationLayer
from infrastructure.ml.anomaly.calibration.profile import (
    DetectorCalibrationProfile,
    compute_calibrated_score,
)
from infrastructure.ml.anomaly.core.detector import VotingAnomalyDetector


class TestCalibrationProfileAndMath:
    """Verifica monotonía, límites, división segura y comportamiento en extremos."""

    def test_range_guarantee(self):
        profile = DetectorCalibrationProfile(tau_nominal=2.0, tau_anomaly=4.0)
        test_inputs = [-100.0, 0.0, 2.0, 3.0, 4.0, 100.0, float("nan"), float("inf"), float("-inf")]
        for x in test_inputs:
            a = compute_calibrated_score(x, profile)
            assert 0.0 <= a <= 1.0
            assert np.isfinite(a)

    def test_monotonicity_standard(self):
        profile = DetectorCalibrationProfile(tau_nominal=2.0, tau_anomaly=5.0)
        scores = np.linspace(0.0, 8.0, 50)
        calibrated = [compute_calibrated_score(s, profile) for s in scores]
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1], f"Violación de monotonía en {scores[i]} -> {scores[i+1]}"

    def test_monotonicity_inverted(self):
        # Isolation Forest / LOF: menor score = mayor anomalía
        profile = DetectorCalibrationProfile(tau_nominal=0.2, tau_anomaly=-0.4, is_inverted=True)
        scores = np.linspace(0.5, -0.8, 50)  # De normal a anómalo
        calibrated = [compute_calibrated_score(s, profile) for s in scores]
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1], "Violación de monotonía en inverted"

    def test_extreme_boundaries(self):
        profile = DetectorCalibrationProfile(tau_nominal=2.0, tau_anomaly=4.0)
        assert compute_calibrated_score(2.0, profile) == 0.0
        assert compute_calibrated_score(1.5, profile) == 0.0
        assert compute_calibrated_score(4.0, profile) == 1.0
        assert compute_calibrated_score(5.5, profile) == 1.0
        assert pytest.approx(compute_calibrated_score(3.0, profile), abs=1e-5) == 0.5

    def test_safe_division_on_identical_thresholds(self):
        # Caso degenerado: tau_nominal == tau_anomaly
        profile = DetectorCalibrationProfile(tau_nominal=2.0, tau_anomaly=2.0)
        assert compute_calibrated_score(1.9, profile) == 0.0
        assert compute_calibrated_score(2.1, profile) == 1.0


class TestCalibrationLayerIsolationAndLookAhead:
    """Verifica aislamiento entre detectores y ausencia de look-ahead."""

    def test_detector_isolation(self):
        layer = AdaptiveDetectorCalibrationLayer()
        layer.set_profile("z_score", DetectorCalibrationProfile(2.0, 4.0))
        layer.set_profile("if", DetectorCalibrationProfile(0.1, -0.3, is_inverted=True))

        assert pytest.approx(layer.transform("z_score", 3.0), abs=1e-5) == 0.5
        assert pytest.approx(layer.transform("if", -0.1), abs=1e-5) == 0.5

        # Modificar z_score no debe alterar "if"
        layer.set_profile("z_score", DetectorCalibrationProfile(1.0, 5.0))
        assert pytest.approx(layer.transform("z_score", 3.0), abs=1e-5) == 0.5
        assert pytest.approx(layer.transform("if", -0.1), abs=1e-5) == 0.5

    def test_train_only_calibration_no_lookahead(self):
        layer = AdaptiveDetectorCalibrationLayer()
        # Solo usamos datos de warm-up nominal (e.g. 100 puntos)
        train_raw_scores = list(np.linspace(0.0, 2.0, 100))
        layer.calibrate_detector("z_score", train_raw_scores, nominal_quantile=0.80, anomaly_quantile=0.98)

        # En streaming (test nunca visto), transform() es causal O(1)
        test_stream = [0.5, 1.7, 1.9, 3.0]
        transformed = [layer.transform("z_score", pt) for pt in test_stream]
        assert transformed[0] < transformed[1] < transformed[2] <= transformed[3]


class TestCalibrationGuardrails:
    """Verifica congelamiento ante drift y bloqueo ante anomalías."""

    def test_guardrail_nominal_and_drift_freeze(self):
        guardrail = CalibrationGuardrail(drift_window=10, persistence_threshold=7)
        assert guardrail.state == CalibrationState.NOMINAL_STABLE
        assert guardrail.is_adaptation_allowed() is True

        # Simular rampa lenta continua (deriva térmica)
        for i in range(12):
            guardrail.observe(value=20.0 + i * 0.5, is_anomaly_active=False)

        assert guardrail.state == CalibrationState.DRIFT_FROZEN
        assert guardrail.is_adaptation_allowed() is False

    def test_guardrail_anomaly_blocked(self):
        guardrail = CalibrationGuardrail()
        guardrail.observe(value=25.0, is_anomaly_active=True)
        assert guardrail.state == CalibrationState.ANOMALY_BLOCKED
        assert guardrail.is_adaptation_allowed() is False


class TestVotingDetectorCalibrationIntegration:
    """Verifica integración y retrocompatibilidad en VotingAnomalyDetector."""

    def test_backward_compatibility_when_layer_none(self):
        # Sin calibration_layer, detector opera idéntico a v1
        det = VotingAnomalyDetector(min_training_points=50)
        train_data = list(np.random.normal(50.0, 2.0, 60))
        det.train(train_data)
        window = SensorWindow(series_id="test_sensor", values=[50.0, 50.1, 50.2], timestamps=[1.0, 2.0, 3.0])
        res = det.detect(window)
        assert res.is_anomaly is False
        assert 0.0 <= res.score <= 1.0

    def test_integration_with_calibration_layer(self):
        layer = AdaptiveDetectorCalibrationLayer()
        det = VotingAnomalyDetector(min_training_points=50, calibration_layer=layer)
        train_data = list(np.random.normal(50.0, 2.0, 60))
        det.train(train_data)

        # Verificar que la capa fue calibrada para los subdetectores
        assert layer.get_profile("z_score") is not None

        # Evaluar punto normal
        window_norm = SensorWindow(series_id="test_sensor", values=[50.0, 50.1], timestamps=[1.0, 2.0])
        res_norm = det.detect(window_norm)
        assert res_norm.score < 0.5

        # Evaluar anomalía severa (outlier extremo)
        window_anom = SensorWindow(series_id="test_sensor", values=[50.0, 150.0], timestamps=[1.0, 2.0])
        res_anom = det.detect(window_anom)
        assert res_anom.score > res_norm.score
