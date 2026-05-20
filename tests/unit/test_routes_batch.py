"""Tests for batch prediction API endpoint."""
import pytest

try:
    import dotenv  # noqa: F401
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

skip_no_dotenv = pytest.mark.skipif(
    not HAS_DOTENV, reason="python-dotenv not installed"
)


@skip_no_dotenv
class TestBatchSchemas:
    """Test batch endpoint schema validation."""

    def test_batch_predict_request_importable(self):
        from iot_machine_learning.ml_service.api.routes_batch import BatchPredictRequest
        assert BatchPredictRequest is not None

    def test_batch_predict_response_importable(self):
        from iot_machine_learning.ml_service.api.routes_batch import BatchPredictResponse
        assert BatchPredictResponse is not None

    def test_batch_item_result_importable(self):
        from iot_machine_learning.ml_service.api.routes_batch import BatchPredictItemResult
        assert BatchPredictItemResult is not None

    def test_batch_request_validation_min(self):
        from iot_machine_learning.ml_service.api.routes_batch import BatchPredictRequest
        from iot_machine_learning.ml_service.api.schemas import PredictRequest

        req = BatchPredictRequest(
            predictions=[PredictRequest(sensor_id=1)],
            max_concurrency=5,
        )
        assert req.max_concurrency == 5
        assert len(req.predictions) == 1

    def test_batch_request_max_concurrency_default(self):
        from iot_machine_learning.ml_service.api.routes_batch import BatchPredictRequest
        from iot_machine_learning.ml_service.api.schemas import PredictRequest

        req = BatchPredictRequest(
            predictions=[PredictRequest(sensor_id=1)],
        )
        assert req.max_concurrency == 10

    def test_batch_request_rejects_empty(self):
        from iot_machine_learning.ml_service.api.routes_batch import BatchPredictRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BatchPredictRequest(predictions=[])

    def test_batch_item_result_success(self):
        from iot_machine_learning.ml_service.api.routes_batch import BatchPredictItemResult

        item = BatchPredictItemResult(
            sensor_id=42, success=True, elapsed_ms=15.3,
        )
        assert item.success is True
        assert item.error is None

    def test_batch_item_result_failure(self):
        from iot_machine_learning.ml_service.api.routes_batch import BatchPredictItemResult

        item = BatchPredictItemResult(
            sensor_id=42, success=False, error="connection timeout", elapsed_ms=3000.0,
        )
        assert item.success is False
        assert "timeout" in item.error

    def test_batch_response_structure(self):
        from iot_machine_learning.ml_service.api.routes_batch import (
            BatchPredictResponse, BatchPredictItemResult,
        )

        resp = BatchPredictResponse(
            total=3, succeeded=2, failed=1,
            results=[
                BatchPredictItemResult(sensor_id=1, success=True, elapsed_ms=10.0),
                BatchPredictItemResult(sensor_id=2, success=True, elapsed_ms=12.0),
                BatchPredictItemResult(sensor_id=3, success=False, error="no data", elapsed_ms=5.0),
            ],
            total_elapsed_ms=25.0,
        )
        assert resp.total == 3
        assert resp.succeeded == 2
        assert resp.failed == 1


@skip_no_dotenv
class TestBatchRouter:
    """Test that the batch router is properly defined."""

    def test_router_exists(self):
        from iot_machine_learning.ml_service.api.routes_batch import router
        assert router is not None

    def test_router_has_batch_endpoint(self):
        from iot_machine_learning.ml_service.api.routes_batch import router
        routes = [r.path for r in router.routes]
        assert "/ml/predict/batch" in routes
