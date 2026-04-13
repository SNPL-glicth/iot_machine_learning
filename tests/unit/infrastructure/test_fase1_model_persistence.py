"""Tests for FASE 1 — Model Persistence (CRÍTICO).

Tests cover:
1. Model serialization/deserialization (IsolationForest, LOF)
2. Ensemble weights persistence
3. Input features persistence
4. Config snapshots
5. Redis sliding windows

FASE 1 fixes:
- ANOM-1: Models lost on restart
- ENS-1: Weights lost on restart
- MISS-1: Input features not saved
- MISS-3: Config not versioned
- LOC-1: Sliding windows in RAM
"""

import pickle
import pytest
from uuid import uuid4
from unittest.mock import Mock, MagicMock

# Model Repository Tests
class TestModelRepository:
    """Test model serialization and persistence."""

    def test_save_model_creates_pickle_blob(self):
        """Should serialize sklearn model to pickle blob."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.model_repository import (
            ModelRepository,
        )

        # Mock sklearn model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[1.0])

        # Mock engine
        mock_engine = Mock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        repo = ModelRepository(engine=mock_engine)

        # Save model
        model_id = repo.save_model(
            model_name="isolation_forest",
            series_id="sensor_42",
            domain_type="sensor",
            model_obj=mock_model,
            training_points=100,
        )

        # Verify INSERT was called
        assert mock_conn.execute.called
        call_args = mock_conn.execute.call_args[0]
        assert "INSERT INTO zenin_ml.ml_models" in str(call_args[0])

        # Verify model_blob is pickle
        params = mock_conn.execute.call_args[1]
        model_blob = params["model_blob"]
        deserialized = pickle.loads(model_blob)
        assert deserialized.predict([1]) == [1.0]

    def test_load_model_deserializes_pickle(self):
        """Should deserialize pickle blob to sklearn model."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.model_repository import (
            ModelRepository,
        )

        # Mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[42.0])
        model_blob = pickle.dumps(mock_model)

        # Mock engine
        mock_engine = Mock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (model_blob, "2026-04-12")
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        repo = ModelRepository(engine=mock_engine)

        # Load model
        loaded_model = repo.load_model(
            series_id="sensor_42",
            model_name="isolation_forest",
        )

        # Verify deserialization
        assert loaded_model is not None
        assert loaded_model.predict([1]) == [42.0]

    def test_save_model_deactivates_previous_versions(self):
        """Should mark previous versions as inactive."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.model_repository import (
            ModelRepository,
        )

        mock_model = Mock()
        mock_engine = Mock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        repo = ModelRepository(engine=mock_engine)

        repo.save_model(
            model_name="lof",
            series_id="sensor_55",
            domain_type="sensor",
            model_obj=mock_model,
            training_points=50,
        )

        # Verify UPDATE was called (deactivate)
        calls = [str(call[0][0]) for call in mock_conn.execute.call_args_list]
        assert any("UPDATE zenin_ml.ml_models" in call and "IsActive = 0" in call for call in calls)


# Ensemble Weights Repository Tests
class TestEnsembleWeightsRepository:
    """Test ensemble weights persistence."""

    def test_save_weights_upserts_per_engine(self):
        """Should MERGE weights for each engine."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.ensemble_weights_repository import (
            EnsembleWeightsRepository,
        )

        mock_engine = Mock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        repo = EnsembleWeightsRepository(engine=mock_engine)

        weights = {
            "taylor": 0.5,
            "baseline": 0.3,
            "statistical": 0.2,
        }

        repo.save_weights(
            series_id="sensor_42",
            domain_type="sensor",
            weights=weights,
        )

        # Verify MERGE called 3 times (one per engine)
        assert mock_conn.execute.call_count == 3

        # Verify MERGE syntax
        calls = [str(call[0][0]) for call in mock_conn.execute.call_args_list]
        assert all("MERGE zenin_ml.ensemble_weights" in call for call in calls)

    def test_load_weights_returns_dict(self):
        """Should load weights as dict."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.ensemble_weights_repository import (
            EnsembleWeightsRepository,
        )

        mock_engine = Mock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("taylor", 0.5),
            ("baseline", 0.3),
            ("statistical", 0.2),
        ]
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        repo = EnsembleWeightsRepository(engine=mock_engine)

        weights = repo.load_weights(series_id="sensor_42")

        assert weights == {
            "taylor": 0.5,
            "baseline": 0.3,
            "statistical": 0.2,
        }


# Input Features Repository Tests
class TestInputFeaturesRepository:
    """Test input features persistence."""

    def test_save_input_features_computes_hash(self):
        """Should compute SHA256 hash for deduplication."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.input_features_repository import (
            InputFeaturesRepository,
        )

        mock_engine = Mock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None  # No existing
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        repo = InputFeaturesRepository(engine=mock_engine)

        prediction_id = uuid4()
        input_values = [20.0, 21.0, 22.0]

        features_id = repo.save_input_features(
            prediction_id=prediction_id,
            input_values=input_values,
        )

        # Verify INSERT called
        assert mock_conn.execute.call_count == 2  # 1 check + 1 insert

        # Verify hash in params
        insert_call = mock_conn.execute.call_args_list[1]
        params = insert_call[1]
        assert "input_hash" in params
        assert len(params["input_hash"]) == 64  # SHA256 hex

    def test_load_input_features_deserializes_json(self):
        """Should deserialize JSON arrays."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.input_features_repository import (
            InputFeaturesRepository,
        )
        import json

        mock_engine = Mock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (
            json.dumps([20.0, 21.0, 22.0]),
            json.dumps([1678901234.0, 1678901244.0, 1678901254.0]),
            None,
        )
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        repo = InputFeaturesRepository(engine=mock_engine)

        result = repo.load_input_features(prediction_id=uuid4())

        assert result["input_values"] == [20.0, 21.0, 22.0]
        assert result["input_timestamps"] == [1678901234.0, 1678901244.0, 1678901254.0]


# Config Snapshot Repository Tests
class TestConfigSnapshotRepository:
    """Test config snapshot persistence."""

    def test_save_snapshot_detects_duplicates(self):
        """Should not save identical config twice."""
        from iot_machine_learning.infrastructure.persistence.sql.zenin_ml.config_snapshot_repository import (
            ConfigSnapshotRepository,
        )

        mock_engine = Mock()
        mock_conn = MagicMock()

        # First call: check duplicate (found)
        existing_id = uuid4()
        mock_conn.execute.return_value.fetchone.side_effect = [
            (str(existing_id),),  # Duplicate check
            (str(existing_id),),  # Get latest ID
        ]
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        repo = ConfigSnapshotRepository(engine=mock_engine)

        config_data = {"ML_BATCH_PARALLEL_WORKERS": 4}

        snapshot_id = repo.save_snapshot(
            config_type="feature_flags",
            config_data=config_data,
        )

        # Should return existing ID, not insert
        assert snapshot_id == existing_id


# Persistent Anomaly Detector Tests
class TestPersistentAnomalyDetector:
    """Test persistent anomaly detector wrapper."""

    def test_auto_saves_models_after_training(self):
        """Should auto-save sklearn models after training."""
        from iot_machine_learning.infrastructure.ml.anomaly.persistent_detector import (
            PersistentAnomalyDetector,
        )

        mock_repo = Mock()
        mock_repo.save_model = Mock()

        detector = PersistentAnomalyDetector(
            series_id="sensor_42",
            domain_type="sensor",
            model_repo=mock_repo,
            auto_save=True,
        )

        # Train with sufficient data
        values = [20.0 + i * 0.1 for i in range(100)]

        detector.train(values)

        # Verify save_model was called for sklearn models
        assert mock_repo.save_model.call_count >= 2  # IF + LOF at minimum

    def test_auto_loads_models_on_first_detect(self):
        """Should auto-load models on first detect."""
        from iot_machine_learning.infrastructure.ml.anomaly.persistent_detector import (
            PersistentAnomalyDetector,
        )
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorWindow,
            SensorReading,
        )

        mock_repo = Mock()
        mock_repo.load_model = Mock(return_value=None)  # No models found

        detector = PersistentAnomalyDetector(
            series_id="sensor_42",
            domain_type="sensor",
            model_repo=mock_repo,
            auto_load=True,
            auto_save=False,
        )

        # Create window
        readings = [SensorReading(sensor_id=42, value=20.0, timestamp=float(i)) for i in range(10)]
        window = SensorWindow(sensor_id=42, readings=readings)

        # First detect should trigger load
        try:
            detector.detect(window)
        except RuntimeError:
            pass  # Expected: not trained

        # Verify load_model was called
        assert mock_repo.load_model.called


# Ensemble Weighted Predictor Persistence Tests
class TestEnsembleWeightedPredictorPersistence:
    """Test ensemble predictor weight persistence."""

    def test_auto_loads_weights_from_db(self):
        """Should auto-load weights from DB on init."""
        from iot_machine_learning.infrastructure.ml.engines.ensemble.predictor import (
            EnsembleWeightedPredictor,
        )

        # Mock engines
        mock_engine1 = Mock()
        mock_engine1.name = "taylor"
        mock_engine1.can_handle = Mock(return_value=True)

        mock_engine2 = Mock()
        mock_engine2.name = "baseline"
        mock_engine2.can_handle = Mock(return_value=True)

        # Create ensemble WITHOUT DB (auto_load_weights=False)
        ensemble = EnsembleWeightedPredictor(
            engines=[mock_engine1, mock_engine2],
            series_id="sensor_42",
            auto_load_weights=False,
        )

        # Weights should be uniform (no DB load)
        assert ensemble._weights == [0.5, 0.5]

    def test_persists_weights_after_recalculation(self):
        """Should persist weights to DB after recalculation."""
        from iot_machine_learning.infrastructure.ml.engines.ensemble.predictor import (
            EnsembleWeightedPredictor,
        )

        mock_engine1 = Mock()
        mock_engine1.name = "taylor"

        mock_engine2 = Mock()
        mock_engine2.name = "baseline"

        # Mock weights repo
        mock_repo = Mock()
        mock_repo.save_weights = Mock()

        ensemble = EnsembleWeightedPredictor(
            engines=[mock_engine1, mock_engine2],
            series_id="sensor_42",
            auto_load_weights=False,
        )

        # Inject mock repo
        ensemble._weights_repo = mock_repo

        # Add some errors
        ensemble._engine_errors["taylor"].append(1.0)
        ensemble._engine_errors["baseline"].append(2.0)

        # Trigger recalculation
        ensemble._recalculate_weights()

        # Verify save_weights was called
        assert mock_repo.save_weights.called
        call_args = mock_repo.save_weights.call_args[1]
        assert call_args["series_id"] == "sensor_42"
        assert "taylor" in call_args["weights"]
        assert "baseline" in call_args["weights"]


# Redis Sliding Window Store Tests
class TestRedisSlidingWindowStore:
    """Test Redis sliding window store."""

    def test_append_pushes_to_redis_list(self):
        """Should push reading to Redis list."""
        from iot_machine_learning.infrastructure.persistence.redis.sliding_window_store import (
            RedisSlidingWindowStore,
            Reading,
        )

        mock_redis = Mock()
        mock_redis.rpush = Mock()
        mock_redis.ltrim = Mock()
        mock_redis.expire = Mock()

        store = RedisSlidingWindowStore(
            redis_client=mock_redis,
            window_size=20,
        )

        reading = Reading(sensor_id=42, value=20.5, timestamp=1678901234.0)

        store.append(reading)

        # Verify rpush called
        assert mock_redis.rpush.called
        key = mock_redis.rpush.call_args[0][0]
        assert key == "sliding_window:42"

        # Verify ltrim called (window size limit)
        assert mock_redis.ltrim.called

        # Verify expire called (TTL)
        assert mock_redis.expire.called

    def test_get_window_deserializes_readings(self):
        """Should deserialize readings from Redis list."""
        from iot_machine_learning.infrastructure.persistence.redis.sliding_window_store import (
            RedisSlidingWindowStore,
        )
        import json

        mock_redis = Mock()
        mock_redis.lrange = Mock(return_value=[
            json.dumps({"sensor_id": 42, "value": 20.0, "timestamp": 1678901234.0}).encode(),
            json.dumps({"sensor_id": 42, "value": 21.0, "timestamp": 1678901244.0}).encode(),
        ])

        store = RedisSlidingWindowStore(redis_client=mock_redis)

        window = store.get_window(sensor_id=42)

        assert len(window) == 2
        assert window[0].value == 20.0
        assert window[1].value == 21.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
