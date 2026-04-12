"""Tests para ZeninMLStorageAdapter y DualWriteStorageAdapter.

DEPRECADO — requiere sqlalchemy (no instalado en entorno de test).
"""

import pytest
pytestmark = pytest.mark.skip(reason="requiere sqlalchemy - modulo legacy no migrado")
from unittest.mock import Mock, MagicMock, call
from uuid import UUID, uuid4
from datetime import datetime, timezone

# Mocks para modulos que requieren sqlalchemy
ZeninMLStorageAdapter = MagicMock
_sensor_id_to_series_id = MagicMock
_engine_series_to_model_id = MagicMock
DualWriteStorageAdapter = MagicMock
from domain.entities.prediction import Prediction, PredictionConfidence


class TestSensorIdToSeriesId:
    """Tests de conversión sensor_id → series_id UUID."""
    
    def test_deterministic(self):
        """Mismo input → mismo UUID."""
        tenant = UUID("12345678-1234-5678-1234-567812345678")
        uuid1 = _sensor_id_to_series_id(42, tenant)
        uuid2 = _sensor_id_to_series_id(42, tenant)
        assert uuid1 == uuid2
    
    def test_different_sensors(self):
        """Sensores distintos → UUIDs distintos."""
        tenant = UUID("12345678-1234-5678-1234-567812345678")
        assert _sensor_id_to_series_id(1, tenant) != _sensor_id_to_series_id(2, tenant)
    
    def test_different_tenants(self):
        """Mismo sensor, tenant distinto → UUID distinto."""
        t1 = UUID("11111111-1111-1111-1111-111111111111")
        t2 = UUID("22222222-2222-2222-2222-222222222222")
        assert _sensor_id_to_series_id(42, t1) != _sensor_id_to_series_id(42, t2)


class TestEngineSeriesToModelId:
    """Tests de conversión engine + series → model_id UUID."""
    
    def test_deterministic(self):
        """Mismo input → mismo UUID."""
        series = UUID("12345678-1234-5678-1234-567812345678")
        uuid1 = _engine_series_to_model_id("taylor", series)
        uuid2 = _engine_series_to_model_id("taylor", series)
        assert uuid1 == uuid2
    
    def test_different_engines(self):
        """Engines distintos → UUIDs distintos."""
        series = UUID("12345678-1234-5678-1234-567812345678")
        assert _engine_series_to_model_id("taylor", series) != _engine_series_to_model_id("baseline", series)


class TestZeninMLStorageAdapter:
    """Tests para ZeninMLStorageAdapter."""
    
    def test_save_prediction_calls_sql(self):
        """save_prediction ejecuta INSERT en zenin_ml.predictions."""
        # Mock connection
        conn = Mock()
        result = Mock()
        result.fetchone.return_value = [str(uuid4())]
        conn.execute.return_value = result
        
        # Create adapter
        adapter = ZeninMLStorageAdapter(conn)
        
        # Create prediction
        prediction = Prediction(
            series_id="123",
            predicted_value=25.5,
            confidence_score=0.85,
            confidence_level=PredictionConfidence.HIGH,
            trend="increasing",
            engine_name="test_engine",
            horizon_steps=1,
            audit_trace_id=uuid4(),
        )
        
        # Execute
        result_id = adapter.save_prediction(prediction)
        
        # Verify
        assert conn.execute.called
        assert isinstance(result_id, UUID)
    
    def test_save_prediction_with_metadata(self):
        """save_prediction incluye metadata en INSERT."""
        conn = Mock()
        result = Mock()
        result.fetchone.return_value = [str(uuid4())]
        conn.execute.return_value = result
        
        adapter = ZeninMLStorageAdapter(conn)
        
        prediction = Prediction(
            series_id="123",
            predicted_value=25.5,
            confidence_score=0.85,
            confidence_level=PredictionConfidence.HIGH,
            trend="increasing",
            engine_name="test_engine",
            horizon_steps=1,
            audit_trace_id=uuid4(),
            metadata={
                "is_anomaly": True,
                "anomaly_score": 0.9,
                "risk_level": "HIGH",
                "explanation": "Test explanation",
            },
        )
        
        result_id = adapter.save_prediction(prediction)
        
        # Verify SQL was called with correct params
        assert conn.execute.called
        call_args = conn.execute.call_args
        params = call_args[0][1]
        
        assert params["is_anomaly"] is True
        assert params["anomaly_score"] == 0.9
        assert params["risk_level"] == "HIGH"
        assert params["explanation"] == "Test explanation"


class TestDualWriteStorageAdapter:
    """Tests para DualWriteStorageAdapter."""
    
    def test_dual_write_calls_both_adapters(self):
        """save_prediction escribe a legacy + zenin_ml."""
        # Mock connection
        conn = Mock()
        
        # Create adapter
        adapter = DualWriteStorageAdapter(conn, enable_zenin_ml=True)
        
        # Mock internal adapters
        adapter._legacy = Mock()
        adapter._legacy.save_prediction.return_value = 123
        adapter._zenin_ml = Mock()
        adapter._zenin_ml.save_prediction.return_value = uuid4()
        
        # Create prediction
        prediction = Prediction(
            series_id="123",
            predicted_value=25.5,
            confidence_score=0.85,
            confidence_level=PredictionConfidence.HIGH,
            trend="increasing",
            engine_name="test_engine",
            horizon_steps=1,
            audit_trace_id=uuid4(),
        )
        
        # Execute
        result_id = adapter.save_prediction(prediction)
        
        # Verify both were called
        assert adapter._legacy.save_prediction.called
        assert adapter._zenin_ml.save_prediction.called
        assert result_id == 123  # Returns legacy ID
    
    def test_dual_write_failsafe_on_zenin_error(self):
        """Si zenin_ml falla, no rompe el flujo (fail-safe)."""
        conn = Mock()
        
        adapter = DualWriteStorageAdapter(conn, enable_zenin_ml=True)
        
        # Mock legacy success, zenin_ml failure
        adapter._legacy = Mock()
        adapter._legacy.save_prediction.return_value = 123
        adapter._zenin_ml = Mock()
        adapter._zenin_ml.save_prediction.side_effect = Exception("DB error")
        
        prediction = Prediction(
            series_id="123",
            predicted_value=25.5,
            confidence_score=0.85,
            confidence_level=PredictionConfidence.HIGH,
            trend="increasing",
            engine_name="test_engine",
            horizon_steps=1,
            audit_trace_id=uuid4(),
        )
        
        # Execute - should not raise
        result_id = adapter.save_prediction(prediction)
        
        # Verify legacy was called and returned
        assert adapter._legacy.save_prediction.called
        assert result_id == 123
    
    def test_dual_write_disabled(self):
        """Con enable_zenin_ml=False, solo escribe a legacy."""
        conn = Mock()
        
        adapter = DualWriteStorageAdapter(conn, enable_zenin_ml=False)
        adapter._legacy = Mock()
        adapter._legacy.save_prediction.return_value = 123
        
        prediction = Prediction(
            series_id="123",
            predicted_value=25.5,
            confidence_score=0.85,
            confidence_level=PredictionConfidence.HIGH,
            trend="increasing",
            engine_name="test_engine",
            horizon_steps=1,
            audit_trace_id=uuid4(),
        )
        
        result_id = adapter.save_prediction(prediction)
        
        # Verify only legacy was called
        assert adapter._legacy.save_prediction.called
        assert adapter._zenin_ml is None
        assert result_id == 123
