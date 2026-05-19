"""Test ML_USE_COGNITIVE_ORCHESTRATOR default value (Phase 3 final)."""

from __future__ import annotations

import pytest


class TestCognitiveOrchestratorDefault:
    """Verify the orchestrator is enabled by default with graceful fallback."""

    def test_cognitive_orchestrator_default_is_true(self) -> None:
        """ML_USE_COGNITIVE_ORCHESTRATOR must be True by default."""
        from iot_machine_learning.ml_service.config.cognitive_config import CognitiveConfig

        config = CognitiveConfig()
        assert config.ML_USE_COGNITIVE_ORCHESTRATOR is True

    def test_cognitive_can_be_explicitly_disabled(self) -> None:
        """Users can opt-out by setting flag to False."""
        from iot_machine_learning.ml_service.config.cognitive_config import CognitiveConfig

        config = CognitiveConfig(ML_USE_COGNITIVE_ORCHESTRATOR=False)
        assert config.ML_USE_COGNITIVE_ORCHESTRATOR is False

    def test_feature_flags_composite_default_is_true(self) -> None:
        """FeatureFlags composite also defaults to True."""
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags

        flags = FeatureFlags()
        assert flags.ML_USE_COGNITIVE_ORCHESTRATOR is True
