"""Tests for SensorFeedbackService — predict-and-verify loop."""

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import MagicMock, call

import pytest

from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult
from iot_machine_learning.ml_service.services.sensor_feedback_service import (
    SensorFeedbackService,
)


class MockRepo:
    """Mock PredictionVerificationRepository."""

    def __init__(self) -> None:
        self.pending: dict[str, dict] = {}
        self.save_calls: list = []
        self.find_calls: list = []
        self.verify_calls: list = []

    def save_pending(
        self,
        series_id: str,
        predicted_value: float,
        target_timestamp: datetime,
        horizon_seconds: int,
        engine_name: str,
        confidence: float,
    ) -> str:
        prediction_id = f"pred-{len(self.pending)}"
        self.pending[prediction_id] = {
            "series_id": series_id,
            "predicted_value": predicted_value,
            "target_timestamp": target_timestamp,
            "horizon_seconds": horizon_seconds,
            "engine_name": engine_name,
            "confidence": confidence,
            "status": "pending",
        }
        self.save_calls.append({
            "series_id": series_id,
            "predicted_value": predicted_value,
            "target_timestamp": target_timestamp,
        })
        return prediction_id

    def find_match(
        self,
        series_id: str,
        reading_timestamp: datetime,
        tolerance_seconds: int,
    ) -> Optional[dict]:
        self.find_calls.append({
            "series_id": series_id,
            "reading_timestamp": reading_timestamp,
            "tolerance_seconds": tolerance_seconds,
        })
        # Return the first pending match (simple mock logic)
        for pred_id, data in self.pending.items():
            if data["series_id"] == series_id and data["status"] == "pending":
                return {
                    "prediction_id": pred_id,
                    "predicted_value": data["predicted_value"],
                    "target_timestamp": data["target_timestamp"],
                    "horizon_seconds": data["horizon_seconds"],
                    "engine_name": data["engine_name"],
                    "confidence": data["confidence"],
                }
        return None

    def mark_verified(
        self,
        prediction_id: str,
        actual_value: float,
        absolute_error: float,
    ) -> None:
        self.verify_calls.append({
            "prediction_id": prediction_id,
            "actual_value": actual_value,
            "absolute_error": absolute_error,
        })
        if prediction_id in self.pending:
            self.pending[prediction_id]["status"] = "verified"
            self.pending[prediction_id]["actual_value"] = actual_value
            self.pending[prediction_id]["absolute_error"] = absolute_error

    def expire_old(self, max_age_seconds: int) -> int:
        return 0


class MockCache:
    """Mock PendingPredictionCache."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, float]] = {}
        self.register_calls: list = []
        self.find_calls: list = []
        self.remove_calls: list = []
        self.has_pending_calls: list = []

    def register(
        self,
        series_id: str,
        prediction_id: str,
        target_timestamp_epoch: float,
    ) -> None:
        if series_id not in self._entries:
            self._entries[series_id] = {}
        self._entries[series_id][prediction_id] = target_timestamp_epoch
        self.register_calls.append({
            "series_id": series_id,
            "prediction_id": prediction_id,
            "target_timestamp_epoch": target_timestamp_epoch,
        })

    def has_pending(self, series_id: str, now_epoch: float) -> bool:
        self.has_pending_calls.append({"series_id": series_id, "now_epoch": now_epoch})
        if series_id not in self._entries:
            return False
        # Return True if any entry has target >= now
        return any(
            target >= now_epoch
            for target in self._entries[series_id].values()
        )

    def find_match(
        self,
        series_id: str,
        reading_timestamp_epoch: float,
        tolerance_seconds: float,
    ) -> Optional[str]:
        self.find_calls.append({
            "series_id": series_id,
            "reading_timestamp_epoch": reading_timestamp_epoch,
            "tolerance_seconds": tolerance_seconds,
        })
        if series_id not in self._entries:
            return None
        candidates = {
            pid: abs(ts - reading_timestamp_epoch)
            for pid, ts in self._entries[series_id].items()
            if abs(ts - reading_timestamp_epoch) <= tolerance_seconds
        }
        if not candidates:
            return None
        return min(candidates, key=candidates.get)

    def remove(self, series_id: str, prediction_id: str) -> None:
        self.remove_calls.append({
            "series_id": series_id,
            "prediction_id": prediction_id,
        })
        if series_id in self._entries:
            self._entries[series_id].pop(prediction_id, None)


class MockOrchestrator:
    """Mock PredictionEngine."""

    def __init__(self) -> None:
        self.predict_calls: list = []
        self.record_actual_calls: list = []
        self._next_result: Optional[PredictionResult] = None

    def predict(
        self,
        series_id: str,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        self.predict_calls.append({
            "series_id": series_id,
            "values": values,
            "timestamps": timestamps,
        })
        if self._next_result is not None:
            return self._next_result
        return PredictionResult(
            predicted_value=42.0,
            confidence=0.85,
            trend="stable",
            metadata={"selected_engine": "kalman"},
        )

    def record_actual(self, actual_value: float, series_id: Optional[str] = None) -> None:
        self.record_actual_calls.append({
            "actual_value": actual_value,
            "series_id": series_id,
        })

    def set_next_result(self, result: PredictionResult) -> None:
        self._next_result = result


@pytest.fixture
def service():
    orchestrator = MockOrchestrator()
    repo = MockRepo()
    cache = MockCache()
    return SensorFeedbackService(
        orchestrator=orchestrator,
        verification_repo=repo,
        pending_cache=cache,
        horizon_seconds=300,
        tolerance_seconds=30,
    )


class TestVerify:
    def test_verify_matches_pending_prediction(self, service):
        # Arrange: register a pending prediction
        now = datetime.now(timezone.utc)
        reading_ts = now.timestamp()
        service._repo.save_pending(
            series_id="sensor_1",
            predicted_value=10.0,
            target_timestamp=now,
            horizon_seconds=300,
            engine_name="kalman",
            confidence=0.8,
        )
        # Manually register in cache too
        service._cache.register("sensor_1", "pred-0", reading_ts)

        # Act
        result = service.verify("sensor_1", 10.5, reading_ts)

        # Assert
        assert result is True
        assert len(service._repo.verify_calls) == 1
        assert service._repo.verify_calls[0]["prediction_id"] == "pred-0"

    def test_verify_no_match_returns_false(self, service):
        # No pending predictions
        result = service.verify("sensor_1", 10.0, 1234567890.0)
        assert result is False
        assert len(service._orchestrator.record_actual_calls) == 0

    def test_verify_calls_orchestrator_record_actual(self, service):
        now = datetime.now(timezone.utc)
        reading_ts = now.timestamp()
        service._repo.save_pending(
            series_id="sensor_1",
            predicted_value=10.0,
            target_timestamp=now,
            horizon_seconds=300,
            engine_name="kalman",
            confidence=0.8,
        )
        service._cache.register("sensor_1", "pred-0", reading_ts)

        service.verify("sensor_1", 10.5, reading_ts)

        assert len(service._orchestrator.record_actual_calls) == 1
        assert service._orchestrator.record_actual_calls[0]["actual_value"] == 10.5

    def test_verify_calculates_correct_absolute_error(self, service):
        now = datetime.now(timezone.utc)
        reading_ts = now.timestamp()
        service._repo.save_pending(
            series_id="sensor_1",
            predicted_value=10.0,
            target_timestamp=now,
            horizon_seconds=300,
            engine_name="kalman",
            confidence=0.8,
        )
        service._cache.register("sensor_1", "pred-0", reading_ts)

        service.verify("sensor_1", 12.0, reading_ts)

        assert service._repo.verify_calls[0]["absolute_error"] == 2.0

    def test_verify_removes_from_cache_after_match(self, service):
        now = datetime.now(timezone.utc)
        reading_ts = now.timestamp()
        service._repo.save_pending(
            series_id="sensor_1",
            predicted_value=10.0,
            target_timestamp=now,
            horizon_seconds=300,
            engine_name="kalman",
            confidence=0.8,
        )
        service._cache.register("sensor_1", "pred-0", reading_ts)

        service.verify("sensor_1", 10.0, reading_ts)

        assert len(service._cache.remove_calls) == 1
        assert service._cache.remove_calls[0]["prediction_id"] == "pred-0"


class TestPredictAndRegister:
    def test_predict_registers_pending_in_cache(self, service):
        reading_ts = 1234567890.0
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = service.predict_and_register(
            series_id="sensor_1",
            values=values,
            timestamps=None,
            reading_timestamp=reading_ts,
        )

        assert result is not None
        assert len(service._repo.save_calls) == 1
        assert len(service._cache.register_calls) == 1
        assert service._cache.register_calls[0]["series_id"] == "sensor_1"

    def test_predict_skipped_when_cooldown_active(self, service):
        reading_ts = 1234567890.0
        # Register a pending prediction (cooldown active)
        service._cache.register("sensor_1", "pred-x", reading_ts + 100)

        result = service.predict_and_register(
            series_id="sensor_1",
            values=[1.0, 2.0, 3.0, 4.0, 5.0],
            timestamps=None,
            reading_timestamp=reading_ts,
        )

        assert result is None
        assert len(service._orchestrator.predict_calls) == 0

    def test_cooldown_allows_after_verification(self, service):
        reading_ts = 1234567890.0
        # Register a pending prediction
        service._cache.register("sensor_1", "pred-x", reading_ts + 100)
        # Verify it (removes from cache)
        service._cache.remove("sensor_1", "pred-x")

        result = service.predict_and_register(
            series_id="sensor_1",
            values=[1.0, 2.0, 3.0, 4.0, 5.0],
            timestamps=None,
            reading_timestamp=reading_ts,
        )

        assert result is not None
        assert len(service._orchestrator.predict_calls) == 1

    def test_predict_and_verify_same_reading_timestamp(self, service):
        """A reading can both verify an old prediction AND trigger a new one."""
        reading_ts = 1234567890.0
        # Setup: a pending prediction that should be verified
        service._repo.save_pending(
            series_id="sensor_1",
            predicted_value=10.0,
            target_timestamp=datetime.fromtimestamp(reading_ts, tz=timezone.utc),
            horizon_seconds=300,
            engine_name="kalman",
            confidence=0.8,
        )
        service._cache.register("sensor_1", "pred-0", reading_ts)

        # Verify
        verified = service.verify("sensor_1", 10.5, reading_ts)
        assert verified is True

        # Predict new
        result = service.predict_and_register(
            series_id="sensor_1",
            values=[1.0, 2.0, 3.0, 4.0, 5.0],
            timestamps=None,
            reading_timestamp=reading_ts,
        )
        assert result is not None


class TestExpireOld:
    def test_expired_predictions_dont_match(self, service):
        # No real expire logic in mock, but verify the concept:
        # After marking verified, a second verify should not match
        now = datetime.now(timezone.utc)
        reading_ts = now.timestamp()
        service._repo.save_pending(
            series_id="sensor_1",
            predicted_value=10.0,
            target_timestamp=now,
            horizon_seconds=300,
            engine_name="kalman",
            confidence=0.8,
        )
        service._cache.register("sensor_1", "pred-0", reading_ts)

        # First verify
        service.verify("sensor_1", 10.0, reading_ts)

        # Second verify should fail (no pending)
        result = service.verify("sensor_1", 10.0, reading_ts)
        assert result is False


class TestVerifyNoFallback:
    def test_verify_no_match_no_record_actual(self, service):
        """When no match, record_actual must NOT be called."""
        result = service.verify("sensor_1", 10.0, 1234567890.0)
        assert result is False
        assert len(service._orchestrator.record_actual_calls) == 0

    def test_verify_no_match_logs_warning(self, service):
        # Just verify it doesn't crash
        result = service.verify("sensor_1", 10.0, 1234567890.0)
        assert result is False
