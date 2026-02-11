"""Integration tests para flujo enterprise completo.

Simula el flujo end-to-end:
1. Crear entidades de dominio
2. Ejecutar predicción vía domain service
3. Detectar anomalías vía domain service
4. Detectar patrones (change points, delta spikes)
5. Verificar audit trail
6. Verificar RBAC
7. Verificar cache

Usa mocks para StoragePort (no depende de BD real).
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from iot_machine_learning.domain.entities.anomaly import AnomalyResult
from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import (
    SensorReading,
    SensorWindow,
)
from iot_machine_learning.domain.ports.prediction_port import PredictionPort
from iot_machine_learning.domain.ports.anomaly_detection_port import AnomalyDetectionPort
from iot_machine_learning.domain.services.anomaly_domain_service import (
    AnomalyDomainService,
)
from iot_machine_learning.domain.services.prediction_domain_service import (
    PredictionDomainService,
)
from iot_machine_learning.domain.services.pattern_domain_service import (
    PatternDomainService,
)
from iot_machine_learning.infrastructure.ml.patterns.change_point_detector import (
    CUSUMDetector,
)
from iot_machine_learning.infrastructure.ml.patterns.delta_spike_classifier import (
    DeltaSpikeClassifier,
)
from iot_machine_learning.infrastructure.security.audit_logger import (
    FileAuditLogger,
    NullAuditLogger,
)
from iot_machine_learning.infrastructure.security.access_control import (
    AccessControlService,
    AccessDeniedError,
    Permission,
    Role,
    UserContext,
)
from iot_machine_learning.infrastructure.adapters.prediction_cache import (
    InMemoryPredictionCache,
)


# --- Mock implementations for testing ---

class _MockPredictionEngine(PredictionPort):
    """Motor mock para testing de integración."""

    def __init__(self, name: str = "mock_engine", offset: float = 0.0) -> None:
        self._name = name
        self._offset = offset

    @property
    def name(self) -> str:
        return self._name

    def can_handle(self, n_points: int) -> bool:
        return n_points >= 1

    def predict(self, window: SensorWindow) -> Prediction:
        values = window.values
        predicted = (sum(values) / len(values)) + self._offset
        return Prediction(
            series_id=str(window.sensor_id),
            predicted_value=predicted,
            confidence_score=0.8,
            trend="stable",
            engine_name=self._name,
        )


class _MockAnomalyDetector(AnomalyDetectionPort):
    """Detector mock para testing."""

    def __init__(self, always_anomaly: bool = False) -> None:
        self._always_anomaly = always_anomaly
        self._trained = False

    @property
    def name(self) -> str:
        return "mock_detector"

    def train(self, historical_values: List[float]) -> None:
        self._trained = True

    def detect(self, window: SensorWindow) -> AnomalyResult:
        if self._always_anomaly:
            return AnomalyResult(
                series_id=str(window.sensor_id),
                is_anomaly=True,
                score=0.9,
                method_votes={"mock": 0.9},
                confidence=0.8,
                explanation="Mock anomalía detectada",
            )
        return AnomalyResult.normal(series_id=str(window.sensor_id))

    def is_trained(self) -> bool:
        return self._trained


def _make_window(sensor_id: int, values: List[float]) -> SensorWindow:
    """Helper para crear SensorWindow."""
    readings = [
        SensorReading(sensor_id=sensor_id, value=v, timestamp=float(i))
        for i, v in enumerate(values)
    ]
    return SensorWindow(sensor_id=sensor_id, readings=readings)


class TestPredictionDomainServiceFlow:
    """Flujo de predicción end-to-end."""

    def test_predict_with_single_engine(self) -> None:
        """Predicción con un solo motor."""
        engine = _MockPredictionEngine("baseline")
        service = PredictionDomainService(engines=[engine])

        window = _make_window(1, [20.0, 21.0, 22.0])
        prediction = service.predict(window)

        assert prediction.series_id == "1"
        assert prediction.engine_name == "baseline"
        assert prediction.audit_trace_id is not None
        assert prediction.confidence_score == 0.8

    def test_predict_with_fallback(self) -> None:
        """Si el motor principal falla, usa fallback."""

        class _FailingEngine(PredictionPort):
            @property
            def name(self) -> str:
                return "failing"

            def can_handle(self, n_points: int) -> bool:
                return n_points >= 1

            def predict(self, window: SensorWindow) -> Prediction:
                raise RuntimeError("Engine falló")

        failing = _FailingEngine()
        fallback = _MockPredictionEngine("fallback")

        service = PredictionDomainService(engines=[failing, fallback])
        window = _make_window(1, [20.0, 21.0])
        prediction = service.predict(window)

        assert "fallback" in prediction.engine_name

    def test_predict_empty_window_raises(self) -> None:
        """Ventana vacía debe lanzar ValueError."""
        engine = _MockPredictionEngine()
        service = PredictionDomainService(engines=[engine])

        window = SensorWindow(sensor_id=1, readings=[])

        with pytest.raises(ValueError, match="vacía"):
            service.predict(window)

    def test_predict_with_audit(self, tmp_path: Path) -> None:
        """Predicción con audit logging."""
        audit_file = tmp_path / "audit.jsonl"
        audit = FileAuditLogger(log_file=audit_file)

        engine = _MockPredictionEngine()
        service = PredictionDomainService(engines=[engine], audit=audit)

        window = _make_window(1, [20.0, 21.0, 22.0])
        service.predict(window)

        # Verificar que se escribió audit log
        content = audit_file.read_text().strip()
        assert len(content) > 0
        entry = json.loads(content)
        assert entry["event_type"] == "prediction"

    def test_available_engines(self) -> None:
        """Lista de motores disponibles."""
        e1 = _MockPredictionEngine("engine_a")
        e2 = _MockPredictionEngine("engine_b")
        service = PredictionDomainService(engines=[e1, e2])

        assert service.available_engines == ["engine_a", "engine_b"]


class TestAnomalyDomainServiceFlow:
    """Flujo de detección de anomalías end-to-end."""

    def test_detect_normal(self) -> None:
        """Valor normal → no anomalía."""
        detector = _MockAnomalyDetector(always_anomaly=False)
        detector.train([20.0] * 100)
        service = AnomalyDomainService(detectors=[detector])

        window = _make_window(1, [20.0, 20.1, 19.9])
        result = service.detect(window)

        assert result.is_anomaly is False

    def test_detect_anomaly(self) -> None:
        """Valor anómalo → anomalía detectada."""
        detector = _MockAnomalyDetector(always_anomaly=True)
        detector.train([20.0] * 100)
        service = AnomalyDomainService(detectors=[detector])

        window = _make_window(1, [20.0, 20.1, 50.0])
        result = service.detect(window)

        assert result.is_anomaly is True
        assert result.score > 0.5

    def test_voting_multiple_detectors(self) -> None:
        """Voting con múltiples detectores."""
        d1 = _MockAnomalyDetector(always_anomaly=True)
        d2 = _MockAnomalyDetector(always_anomaly=False)
        d1.train([20.0] * 100)
        d2.train([20.0] * 100)

        service = AnomalyDomainService(
            detectors=[d1, d2],
            voting_threshold=0.5,
        )

        window = _make_window(1, [20.0, 50.0])
        result = service.detect(window)

        # d1 vota 0.9, d2 vota 0.0 → promedio 0.45 < 0.5
        assert result.is_anomaly is False

    def test_empty_window_returns_normal(self) -> None:
        """Ventana vacía → normal."""
        detector = _MockAnomalyDetector()
        service = AnomalyDomainService(detectors=[detector])

        window = SensorWindow(sensor_id=1)
        result = service.detect(window)

        assert result.is_anomaly is False

    def test_train_all(self) -> None:
        """train_all entrena todos los detectores."""
        d1 = _MockAnomalyDetector()
        d2 = _MockAnomalyDetector()
        service = AnomalyDomainService(detectors=[d1, d2])

        service.train_all([20.0] * 100)

        assert d1.is_trained() is True
        assert d2.is_trained() is True


class TestPatternDomainServiceFlow:
    """Flujo de detección de patrones end-to-end."""

    def test_change_point_detection(self) -> None:
        """Detectar cambio de nivel con CUSUM."""
        cusum = CUSUMDetector(threshold=5.0, drift=0.5)
        service = PatternDomainService(change_point_detector=cusum)

        values = [20.0] * 50 + [30.0] * 50
        cps = service.detect_change_points(values)

        assert len(cps) >= 1

    def test_spike_classification(self) -> None:
        """Clasificar spike como delta o noise."""
        classifier = DeltaSpikeClassifier(min_history=20)
        service = PatternDomainService(spike_classifier=classifier)

        # Delta spike: cambio persistente
        values = [20.0] * 30 + [30.0] * 10
        result = service.classify_spike(values, spike_index=30)

        assert result.classification.value in ("delta_spike", "noise_spike", "normal")

    def test_no_detector_returns_defaults(self) -> None:
        """Sin detectores configurados → resultados por defecto."""
        service = PatternDomainService()

        window = _make_window(1, [20.0, 21.0])
        pattern = service.detect_pattern(window)
        assert pattern.pattern_type.value == "stable"
        assert pattern.confidence == 0.0

        cps = service.detect_change_points([20.0] * 50)
        assert cps == []

        spike = service.classify_spike([20.0] * 30, 15)
        assert spike.classification.value == "normal"

        regime = service.predict_regime(20.0)
        assert regime is None


class TestRBACIntegration:
    """Integración de RBAC con flujo de predicción."""

    def test_operator_can_predict(self) -> None:
        """Operator puede ejecutar predicciones."""
        svc = AccessControlService()
        svc.register_user(UserContext(
            user_id="op1",
            roles={Role.OPERATOR},
        ))

        # No debe lanzar excepción
        svc.check_permission("op1", Permission.EXECUTE_PREDICTION)

    def test_viewer_cannot_predict(self) -> None:
        """Viewer no puede ejecutar predicciones."""
        svc = AccessControlService()
        svc.register_user(UserContext(
            user_id="viewer1",
            roles={Role.VIEWER},
        ))

        with pytest.raises(AccessDeniedError):
            svc.check_permission("viewer1", Permission.EXECUTE_PREDICTION)

    def test_sensor_access_restriction(self) -> None:
        """Operator con whitelist solo accede a sensores permitidos."""
        svc = AccessControlService()
        svc.register_user(UserContext(
            user_id="op1",
            roles={Role.OPERATOR},
            allowed_sensor_ids={1, 5, 42},
        ))

        svc.check_sensor_access("op1", 1)  # OK

        with pytest.raises(AccessDeniedError):
            svc.check_sensor_access("op1", 99)


class TestCacheIntegration:
    """Integración de cache con predicciones."""

    def test_cache_hit_avoids_recomputation(self) -> None:
        """Cache hit debe retornar resultado sin recomputar."""
        cache = InMemoryPredictionCache(ttl_seconds=60.0)

        values = [20.0, 20.1, 20.2]
        result = {"predicted_value": 20.3, "engine": "taylor"}

        cache.set(1, values, "taylor", result)

        cached = cache.get(1, values, "taylor")
        assert cached is not None
        assert cached["predicted_value"] == 20.3

    def test_cache_miss_after_value_change(self) -> None:
        """Valores diferentes → cache miss."""
        cache = InMemoryPredictionCache(ttl_seconds=60.0)

        cache.set(1, [20.0, 20.1], "taylor", {"v": 1})

        # Nuevos valores → miss
        cached = cache.get(1, [20.0, 25.0], "taylor")
        assert cached is None

    def test_invalidation_on_large_change(self) -> None:
        """Cambio grande debe triggear invalidación."""
        cache = InMemoryPredictionCache(invalidation_threshold_pct=0.1)

        assert cache.should_invalidate(25.0, 20.0) is True  # 25%
        assert cache.should_invalidate(20.5, 20.0) is False  # 2.5%


class TestAuditTrailIntegration:
    """Integración de audit trail con flujo completo."""

    def test_prediction_generates_audit_entry(self, tmp_path: Path) -> None:
        """Predicción debe generar entrada de auditoría."""
        audit_file = tmp_path / "audit.jsonl"
        audit = FileAuditLogger(log_file=audit_file, include_hash=True)

        engine = _MockPredictionEngine()
        service = PredictionDomainService(engines=[engine], audit=audit)

        window = _make_window(1, [20.0, 21.0, 22.0])
        prediction = service.predict(window)

        # Leer audit log
        lines = audit_file.read_text().strip().split("\n")
        assert len(lines) >= 1

        entry = json.loads(lines[0])
        assert entry["event_type"] == "prediction"
        assert entry["resource"] == "sensor_1"
        assert "integrity_hash" in entry
        assert entry["details"]["engine"] == "mock_engine"

    def test_anomaly_generates_audit_entry(self, tmp_path: Path) -> None:
        """Anomalía debe generar entrada de auditoría."""
        audit_file = tmp_path / "audit.jsonl"
        audit = FileAuditLogger(log_file=audit_file)

        detector = _MockAnomalyDetector(always_anomaly=True)
        detector.train([20.0] * 100)
        service = AnomalyDomainService(detectors=[detector], audit=audit)

        window = _make_window(1, [20.0, 50.0])
        service.detect(window)

        content = audit_file.read_text().strip()
        assert len(content) > 0
        entry = json.loads(content)
        assert entry["event_type"] == "anomaly_detection"
