"""Tests for Weaviate batch operations."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from iot_machine_learning.infrastructure.adapters.weaviate.batch_operations import (
    WeaviateBatch,
    BatchResult,
)


class TestBatchResult:
    """Tests for BatchResult dataclass."""
    
    def test_success_rate_calculation(self):
        """Test success rate percentage calculation."""
        result = BatchResult(total=100, successful=85, failed=15)
        assert result.success_rate == 85.0
    
    def test_success_rate_zero_total(self):
        """Test success rate with zero total (avoid division by zero)."""
        result = BatchResult(total=0, successful=0, failed=0)
        assert result.success_rate == 0.0
    
    def test_to_dict(self):
        """Test conversion to dict for logging."""
        result = BatchResult(
            total=50,
            successful=48,
            failed=2,
            errors=["error1", "error2"],
        )
        d = result.to_dict()
        assert d["total"] == 50
        assert d["successful"] == 48
        assert d["failed"] == 2
        assert d["success_rate"] == 96.0
        assert d["error_count"] == 2


class TestWeaviateBatch:
    """Tests for WeaviateBatch class."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        batch = WeaviateBatch("http://localhost:8080")
        assert batch._base_url == "http://localhost:8080"
        assert batch._batch_size == 100
        assert batch._enabled is True
        assert batch._dry_run is False
        assert batch.pending_count == 0
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        batch = WeaviateBatch(
            "http://localhost:8080/",
            batch_size=50,
            enabled=False,
            dry_run=True,
            timeout=30,
        )
        assert batch._base_url == "http://localhost:8080"
        assert batch._batch_size == 50
        assert batch._enabled is False
        assert batch._dry_run is True
        assert batch._timeout == 30
    
    def test_add_object_accumulates(self):
        """Test that add_object accumulates objects."""
        batch = WeaviateBatch("http://localhost:8080", batch_size=10)
        
        batch.add_object("MLExplanation", {"text": "test1"})
        assert batch.pending_count == 1
        
        batch.add_object("MLAnomaly", {"score": 0.9})
        assert batch.pending_count == 2
    
    def test_add_object_disabled(self):
        """Test that add_object does nothing when disabled."""
        batch = WeaviateBatch("http://localhost:8080", enabled=False)
        batch.add_object("MLExplanation", {"text": "test"})
        assert batch.pending_count == 0
    
    @patch("iot_machine_learning.infrastructure.adapters.weaviate.batch_operations.post_json")
    def test_auto_flush_on_batch_size(self, mock_post):
        """Test that batch auto-flushes when batch_size is reached."""
        mock_post.return_value = [
            {"result": {"status": "SUCCESS"}, "id": f"uuid-{i}"}
            for i in range(5)
        ]
        
        batch = WeaviateBatch("http://localhost:8080", batch_size=5)
        
        # Add 4 objects - should not flush
        for i in range(4):
            batch.add_object("MLExplanation", {"text": f"test{i}"})
        assert batch.pending_count == 4
        assert mock_post.call_count == 0
        
        # Add 5th object - should auto-flush
        batch.add_object("MLExplanation", {"text": "test4"})
        assert batch.pending_count == 0
        assert mock_post.call_count == 1
    
    def test_flush_empty_batch(self):
        """Test flushing an empty batch returns empty result."""
        batch = WeaviateBatch("http://localhost:8080")
        result = batch.flush()
        assert result.total == 0
        assert result.successful == 0
        assert result.failed == 0
    
    def test_flush_dry_run(self):
        """Test dry run mode logs but doesn't send."""
        batch = WeaviateBatch("http://localhost:8080", dry_run=True)
        batch.add_object("MLExplanation", {"text": "test"})
        
        result = batch.flush()
        assert result.total == 1
        assert result.successful == 1
        assert result.uuids == ["dry-run-uuid"]
        assert batch.pending_count == 0
    
    @patch("iot_machine_learning.infrastructure.adapters.weaviate.batch_operations.post_json")
    def test_flush_success(self, mock_post):
        """Test successful batch flush."""
        mock_post.return_value = [
            {"result": {"status": "SUCCESS"}, "id": "uuid-1"},
            {"result": {"status": "SUCCESS"}, "id": "uuid-2"},
        ]
        
        batch = WeaviateBatch("http://localhost:8080")
        batch.add_object("MLExplanation", {"text": "test1"})
        batch.add_object("MLAnomaly", {"score": 0.9})
        
        result = batch.flush()
        assert result.total == 2
        assert result.successful == 2
        assert result.failed == 0
        assert len(result.uuids) == 2
        assert batch.pending_count == 0
    
    @patch("iot_machine_learning.infrastructure.adapters.weaviate.batch_operations.post_json")
    def test_flush_partial_failure(self, mock_post):
        """Test batch flush with some failures."""
        mock_post.return_value = [
            {"result": {"status": "SUCCESS"}, "id": "uuid-1"},
            {
                "result": {
                    "status": "FAILED",
                    "errors": {"error": [{"message": "Invalid property"}]},
                },
            },
        ]
        
        batch = WeaviateBatch("http://localhost:8080")
        batch.add_object("MLExplanation", {"text": "test1"})
        batch.add_object("MLAnomaly", {"invalid": "data"})
        
        result = batch.flush()
        assert result.total == 2
        assert result.successful == 1
        assert result.failed == 1
        assert len(result.errors) == 1
        assert "Invalid property" in result.errors[0]
    
    @patch("iot_machine_learning.infrastructure.adapters.weaviate.batch_operations.post_json")
    def test_flush_http_error(self, mock_post):
        """Test batch flush with HTTP error."""
        mock_post.side_effect = Exception("Connection refused")
        
        batch = WeaviateBatch("http://localhost:8080")
        batch.add_object("MLExplanation", {"text": "test"})
        
        result = batch.flush()
        assert result.total == 1
        assert result.successful == 0
        assert result.failed == 1
        assert len(result.errors) == 1
        assert "Connection refused" in result.errors[0]
    
    def test_get_class_distribution(self):
        """Test class distribution calculation."""
        batch = WeaviateBatch("http://localhost:8080")
        batch.add_object("MLExplanation", {"text": "test1"})
        batch.add_object("MLExplanation", {"text": "test2"})
        batch.add_object("MLAnomaly", {"score": 0.9})
        
        dist = batch._get_class_distribution()
        assert dist["MLExplanation"] == 2
        assert dist["MLAnomaly"] == 1
    
    @patch("iot_machine_learning.infrastructure.adapters.weaviate.batch_operations.post_json")
    def test_statistics_tracking(self, mock_post):
        """Test that statistics are tracked across multiple flushes."""
        mock_post.return_value = [
            {"result": {"status": "SUCCESS"}, "id": f"uuid-{i}"}
            for i in range(3)
        ]
        
        batch = WeaviateBatch("http://localhost:8080", batch_size=3)
        
        # First batch
        for i in range(3):
            batch.add_object("MLExplanation", {"text": f"test{i}"})
        # Auto-flushed
        
        # Second batch
        for i in range(3):
            batch.add_object("MLAnomaly", {"score": i * 0.1})
        # Auto-flushed
        
        assert batch.total_sent == 6
        assert batch.total_successful == 6
        assert batch.total_failed == 0
        assert batch.overall_success_rate == 100.0
    
    @patch("iot_machine_learning.infrastructure.adapters.weaviate.batch_operations.post_json")
    def test_context_manager(self, mock_post):
        """Test context manager auto-flushes on exit."""
        mock_post.return_value = [
            {"result": {"status": "SUCCESS"}, "id": "uuid-1"},
        ]
        
        with WeaviateBatch("http://localhost:8080") as batch:
            batch.add_object("MLExplanation", {"text": "test"})
            assert batch.pending_count == 1
        
        # Should have flushed on exit
        assert mock_post.call_count == 1
    
    def test_get_stats(self):
        """Test get_stats returns comprehensive statistics."""
        batch = WeaviateBatch("http://localhost:8080")
        batch.add_object("MLExplanation", {"text": "test"})
        batch._total_sent = 100
        batch._total_successful = 95
        batch._total_failed = 5
        
        stats = batch.get_stats()
        assert stats["pending"] == 1
        assert stats["total_sent"] == 100
        assert stats["total_successful"] == 95
        assert stats["total_failed"] == 5
        assert stats["success_rate"] == 95.0


class TestBatchPropertyBuilders:
    """Tests for batch property builder functions."""
    
    def test_build_explanation_properties(self):
        """Test building explanation properties for batch."""
        from iot_machine_learning.domain.entities.prediction import (
            Prediction,
            PredictionConfidence,
        )
        from iot_machine_learning.infrastructure.adapters.weaviate.memory_writers import (
            build_explanation_properties,
        )
        
        prediction = Prediction(
            series_id="sensor-123",
            predicted_value=25.5,
            confidence_score=0.85,
            confidence_level=PredictionConfidence.HIGH,
            trend="stable",
            engine_name="taylor",
            horizon_steps=10,
            metadata={"explanation": "Test explanation"},
        )
        
        props = build_explanation_properties(prediction, source_record_id=42)
        
        assert props["seriesId"] == "sensor-123"
        assert props["predictedValue"] == 25.5
        assert props["confidenceScore"] == 0.85
        assert props["trend"] == "stable"
        assert props["engineName"] == "taylor"
        assert props["sourceRecordId"] == 42
        assert "Test explanation" in props["explanationText"]
    
    def test_build_anomaly_properties(self):
        """Test building anomaly properties for batch."""
        from iot_machine_learning.domain.entities.anomaly import (
            AnomalyResult,
            AnomalySeverity,
        )
        from iot_machine_learning.infrastructure.adapters.weaviate.memory_writers import (
            build_anomaly_properties,
        )
        
        anomaly = AnomalyResult(
            series_id="sensor-456",
            is_anomaly=True,
            anomaly_score=0.92,
            anomaly_confidence=0.88,
            severity=AnomalySeverity.HIGH,
            detection_methods=["zscore", "lof"],
            method_votes={"zscore": True, "lof": True},
        )
        
        props = build_anomaly_properties(anomaly, source_record_id=99)
        
        assert props["seriesId"] == "sensor-456"
        assert props["anomalyScore"] == 0.92
        assert props["anomalyConfidence"] == 0.88
        assert props["severity"] == "HIGH"
        assert props["sourceRecordId"] == 99
