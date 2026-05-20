"""Coverage tests for ml_service/config/ — Pydantic config models.

Covers: batch_config, cognitive_config, decision_config, feature_flags,
flags, loader, ml_config, parsers, security_config, taylor_config,
threshold_config
"""
from __future__ import annotations

import pytest


class TestBatchConfig:
    def test_defaults(self):
        from iot_machine_learning.ml_service.config.batch_config import BatchConfig
        c = BatchConfig()
        assert c.ML_BATCH_MAX_WORKERS == 4
        assert c.ML_BATCH_PARALLEL_WORKERS == 8
        assert c.ML_MQTT_NUM_WORKERS == 4
        assert c.ML_SLIDING_WINDOW_MAX_SENSORS == 1000

    def test_overrides(self):
        from iot_machine_learning.ml_service.config.batch_config import BatchConfig
        c = BatchConfig(ML_BATCH_MAX_WORKERS=16, ML_BATCH_PARALLEL_WORKERS=32)
        assert c.ML_BATCH_MAX_WORKERS == 16
        assert c.ML_BATCH_PARALLEL_WORKERS == 32


class TestCognitiveConfig:
    def test_importable(self):
        from iot_machine_learning.ml_service.config import cognitive_config
        assert cognitive_config is not None


class TestDecisionConfig:
    def test_importable(self):
        from iot_machine_learning.ml_service.config import decision_config
        assert decision_config is not None

    def test_has_bayes_ttl(self):
        from iot_machine_learning.ml_service.config.decision_config import DecisionConfig
        c = DecisionConfig()
        assert c.ML_BAYES_REDIS_CACHE_TTL_SECONDS == 60.0


class TestFeatureFlags:
    def test_importable(self):
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        assert FeatureFlags is not None

    def test_defaults(self):
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        assert hasattr(flags, "ML_USE_COGNITIVE_ORCHESTRATOR")


class TestFlags:
    def test_importable(self):
        from iot_machine_learning.ml_service.config import flags
        assert flags is not None


class TestLoader:
    def test_importable(self):
        from iot_machine_learning.ml_service.config import loader
        assert loader is not None

    def test_get_feature_flags(self):
        from iot_machine_learning.ml_service.config.loader import get_feature_flags
        flags = get_feature_flags()
        assert flags is not None


class TestMLConfig:
    def test_importable(self):
        from iot_machine_learning.ml_service.config.ml_config import GlobalMLConfig
        assert GlobalMLConfig is not None


class TestParsers:
    def test_importable(self):
        from iot_machine_learning.ml_service.config import parsers
        assert parsers is not None


class TestSecurityConfig:
    def test_importable(self):
        from iot_machine_learning.ml_service.config import security_config
        assert security_config is not None


class TestTaylorConfig:
    def test_importable(self):
        from iot_machine_learning.ml_service.config import taylor_config
        assert taylor_config is not None


class TestThresholdConfig:
    def test_importable(self):
        from iot_machine_learning.ml_service.config import threshold_config
        assert threshold_config is not None
