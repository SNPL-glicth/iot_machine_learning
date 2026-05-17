"""Tests para Sprint 2: MoEPredictionEngine, ContextualRegimeGating, DiscrepancyAwareFusion.

Tests obligatorios:
- test_moe_engine_predict: MoEPredictionEngine retorna PredictionResult válida.
- test_moe_fusion_penalizes_outliers: discrepancia alta → confianza < 0.7.
- test_moe_gating_uses_numeric_features: usa std/slope, no solo regime string.
- test_moe_fallback_on_empty_registry: fallback sin crash.
"""

from unittest.mock import MagicMock

import pytest

from iot_machine_learning.infrastructure.ml.moe import (
    MoEPredictionEngine,
    ContextualRegimeGating,
    DiscrepancyAwareFusion,
    FeatureContext,
    ExpertRegistry,
)
from iot_machine_learning.infrastructure.ml.moe.gating.base import GatingProbs
from iot_machine_learning.domain.ports.expert_port import ExpertOutput, ExpertCapability
from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.iot.sensor_reading import SensorWindow, Reading


class TestMoEPredictionEnginePredict:
    """test_moe_engine_predict — MoEPredictionEngine retorna Prediction válida."""

    def _make_engine(self, registry=None, fallback=None):
        registry = registry or ExpertRegistry()
        gating = ContextualRegimeGating(expert_ids=registry.list_all())
        fusion = DiscrepancyAwareFusion()
        return MoEPredictionEngine(
            registry=registry,
            gating=gating,
            fusion=fusion,
            fallback_engine=fallback,
            sparsity_k=2,
        )

    def test_predict_with_feature_context_returns_valid_prediction(self):
        """Cuando FeatureContext es proveído, retorna PredictionResult sin recalcular."""
        registry = ExpertRegistry()
        mock_expert = MagicMock()
        mock_expert.can_handle.return_value = True
        mock_expert.predict.return_value = ExpertOutput(
            prediction=42.0, confidence=0.85, trend="stable"
        )
        mock_expert.capabilities = ExpertCapability(
            regimes=("stable",), min_points=1
        )
        registry.register("baseline", mock_expert, mock_expert.capabilities)

        engine = self._make_engine(registry=registry)
        ctx = FeatureContext(
            regime="stable",
            mean=40.0,
            std=2.0,
            slope=0.1,
            curvature=0.0,
            noise_ratio=0.05,
            stability=0.9,
            hampel_outlier_mask=[],
            spatial_correlation_score=0.0,
        )
        values = [40.0, 41.0, 42.0, 43.0]

        result = engine.predict_with_context(values, None, ctx)

        assert result is not None
        assert result.predicted_value is not None
        assert 0.0 <= result.confidence <= 1.0
        assert result.trend in ("up", "down", "stable")

    def test_predict_standalone_builds_basic_context(self):
        """Sin FeatureContext, construye uno básico y funciona."""
        registry = ExpertRegistry()
        mock_expert = MagicMock()
        mock_expert.can_handle.return_value = True
        mock_expert.predict.return_value = ExpertOutput(
            prediction=10.0, confidence=0.8, trend="up"
        )
        mock_expert.capabilities = ExpertCapability(
            regimes=("stable", "trending"), min_points=1
        )
        registry.register("baseline", mock_expert, mock_expert.capabilities)

        engine = self._make_engine(registry=registry)
        values = [8.0, 9.0, 10.0, 11.0]

        result = engine.predict(values)

        assert result is not None
        assert result.predicted_value is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_name_is_moe_engine(self):
        engine = self._make_engine()
        assert engine.name == "moe_engine"

    def test_as_port_returns_prediction_port(self):
        engine = self._make_engine()
        port = engine.as_port()
        assert port is not None
        assert port.name == "moe_engine"


class TestDiscrepancyAwareFusion:
    """test_moe_fusion_penalizes_outliers — discrepancia alta penaliza confianza."""

    def test_high_discrepancy_reduces_confidence(self):
        """Dos expertos predicen 20 y 80 con confianza 0.9 → fused < 0.7."""
        fusion = DiscrepancyAwareFusion(discrepancy_threshold=2.0)

        expert_outputs = {
            "expert_a": ExpertOutput(prediction=20.0, confidence=0.9, trend="stable"),
            "expert_b": ExpertOutput(prediction=80.0, confidence=0.9, trend="stable"),
        }
        weights = {"expert_a": 0.5, "expert_b": 0.5}

        result = fusion.fuse(expert_outputs, weights)

        assert result.confidence_score < 0.7, (
            f"Expected fused confidence < 0.7, got {result.confidence_score}"
        )
        # También verificar que nunca supera max expert confidence
        assert result.confidence_score <= 0.9

    def test_low_discrepancy_keeps_confidence(self):
        """Expertos de acuerdo → confianza no penalizada."""
        fusion = DiscrepancyAwareFusion(discrepancy_threshold=2.0)

        expert_outputs = {
            "expert_a": ExpertOutput(prediction=50.0, confidence=0.9, trend="stable"),
            "expert_b": ExpertOutput(prediction=51.0, confidence=0.85, trend="stable"),
        }
        weights = {"expert_a": 0.5, "expert_b": 0.5}

        result = fusion.fuse(expert_outputs, weights)

        # Confianza ponderada sin penalización ≈ 0.875
        assert result.confidence_score > 0.8
        assert result.confidence_score <= 0.9

    def test_fused_value_is_weighted_average(self):
        fusion = DiscrepancyAwareFusion()
        expert_outputs = {
            "a": ExpertOutput(prediction=10.0, confidence=0.8, trend="up"),
            "b": ExpertOutput(prediction=20.0, confidence=0.6, trend="up"),
        }
        weights = {"a": 0.5, "b": 0.5}

        result = fusion.fuse(expert_outputs, weights)
        assert result.predicted_value == 15.0

    def test_empty_outputs_raises(self):
        fusion = DiscrepancyAwareFusion()
        with pytest.raises(ValueError, match="No hay expert_outputs"):
            fusion.fuse({}, {})


class TestContextualRegimeGating:
    """test_moe_gating_uses_numeric_features — usa std/slope, no solo regime."""

    def _make_gating(self, expert_ids=None):
        return ContextualRegimeGating(
            expert_ids=expert_ids or ["baseline", "statistical", "taylor", "kalman"],
        )

    def test_uses_std_to_boost_volatile_experts(self):
        """Alta std debe aumentar peso de taylor/kalman incluso en regime='stable'."""
        gating = self._make_gating()

        # Mismo regime, diferente std
        ctx_low_std = FeatureContext(
            regime="stable", mean=10.0, std=0.1, slope=0.0,
            curvature=0.0, noise_ratio=0.05, stability=0.9,
            hampel_outlier_mask=[], spatial_correlation_score=0.0,
        )
        ctx_high_std = FeatureContext(
            regime="stable", mean=10.0, std=5.0, slope=0.0,
            curvature=0.0, noise_ratio=0.05, stability=0.9,
            hampel_outlier_mask=[], spatial_correlation_score=0.0,
        )

        probs_low = gating.route(ctx_low_std)
        probs_high = gating.route(ctx_high_std)

        # En alta std, taylor debe tener más peso relativo
        # (kalman empieza en 0.0 para stable, no puede boostearse desde 0)
        assert probs_high.probabilities["taylor"] > probs_low.probabilities["taylor"]

    def test_uses_slope_to_boost_statistical(self):
        """Slope alto debe aumentar peso de statistical."""
        gating = self._make_gating()

        ctx_no_slope = FeatureContext(
            regime="stable", mean=10.0, std=1.0, slope=0.0,
            curvature=0.0, noise_ratio=0.1, stability=0.9,
            hampel_outlier_mask=[], spatial_correlation_score=0.0,
        )
        ctx_high_slope = FeatureContext(
            regime="stable", mean=10.0, std=1.0, slope=2.0,
            curvature=0.0, noise_ratio=0.1, stability=0.9,
            hampel_outlier_mask=[], spatial_correlation_score=0.0,
        )

        probs_no = gating.route(ctx_no_slope)
        probs_high = gating.route(ctx_high_slope)

        assert probs_high.probabilities["statistical"] > probs_no.probabilities["statistical"]

    def test_probabilities_sum_to_one(self):
        gating = self._make_gating()
        ctx = FeatureContext(
            regime="trending", mean=10.0, std=1.5, slope=0.5,
            curvature=0.0, noise_ratio=0.1, stability=0.7,
            hampel_outlier_mask=[], spatial_correlation_score=0.0,
        )
        probs = gating.route(ctx)
        total = sum(probs.probabilities.values())
        assert 0.99 <= total <= 1.01

    def test_explain_returns_human_readable_string(self):
        gating = self._make_gating()
        ctx = FeatureContext(
            regime="volatile", mean=10.0, std=3.0, slope=0.5,
            curvature=0.0, noise_ratio=0.4, stability=0.3,
            hampel_outlier_mask=[], spatial_correlation_score=0.0,
        )
        probs = gating.route(ctx)
        explanation = gating.explain(ctx, probs)

        assert isinstance(explanation, str)
        assert "top_expert=" in explanation
        assert "régimen=volatile" in explanation
        assert "Entropía=" in explanation

    def test_deterministic_output(self):
        """Mismo input → mismo output (determinista)."""
        gating = self._make_gating()
        ctx = FeatureContext(
            regime="stable", mean=10.0, std=0.5, slope=0.0,
            curvature=0.0, noise_ratio=0.05, stability=0.95,
            hampel_outlier_mask=[], spatial_correlation_score=0.0,
        )
        probs1 = gating.route(ctx)
        probs2 = gating.route(ctx)

        assert probs1.probabilities == probs2.probabilities
        assert probs1.top_expert == probs2.top_expert


class TestMoEFallbackOnEmptyRegistry:
    """test_moe_fallback_on_empty_registry — fallback sin crash."""

    def test_fallback_when_no_experts_available(self):
        """Sin expertos, usa fallback_engine."""
        fallback = MagicMock()
        fallback.predict.return_value = Prediction(
            series_id="test",
            predicted_value=99.0,
            confidence_score=0.5,
            trend="stable",
            engine_name="fallback",
            metadata={},
        )

        registry = ExpertRegistry()  # vacío
        engine = MoEPredictionEngine(
            registry=registry,
            fallback_engine=fallback,
            sparsity_k=2,
        )

        values = [1.0, 2.0, 3.0]
        result = engine.predict(values)

        assert result is not None
        assert result.predicted_value == 99.0
        fallback.predict.assert_called_once()

    def test_fallback_metadata_indicates_reason(self):
        fallback = MagicMock()
        fallback.predict.return_value = Prediction(
            series_id="test",
            predicted_value=50.0,
            confidence_score=0.6,
            trend="stable",
            engine_name="fallback",
            metadata={},
        )

        registry = ExpertRegistry()
        engine = MoEPredictionEngine(
            registry=registry,
            fallback_engine=fallback,
            sparsity_k=2,
        )

        result = engine.predict([1.0, 2.0, 3.0])
        assert result.metadata.get("moe_fallback") is True
        assert "fallback_reason" in result.metadata

    def test_can_handle_with_empty_registry_and_fallback(self):
        fallback = MagicMock()
        fallback.can_handle.return_value = True

        registry = ExpertRegistry()
        engine = MoEPredictionEngine(
            registry=registry,
            fallback_engine=fallback,
            sparsity_k=2,
        )

        assert engine.can_handle(5) is True

    def test_can_handle_without_fallback_returns_false(self):
        registry = ExpertRegistry()
        engine = MoEPredictionEngine(
            registry=registry,
            fallback_engine=None,
            sparsity_k=2,
        )

        assert engine.can_handle(5) is False


class TestShadowGating:
    """test_shadow_gating_does_not_affect_routing — TreeGating shadow no cambia resultado."""

    def test_shadow_gating_does_not_affect_routing(self):
        """TreeGatingNetwork en shadow no altera la predicción final."""
        from iot_machine_learning.infrastructure.ml.moe.gating.tree_gating import TreeGatingNetwork

        registry = ExpertRegistry()
        mock_expert = MagicMock()
        mock_expert.can_handle.return_value = True
        mock_expert.predict.return_value = ExpertOutput(
            prediction=42.0, confidence=0.85, trend="stable"
        )
        mock_expert.capabilities = ExpertCapability(
            regimes=("stable",), min_points=1
        )
        registry.register("baseline", mock_expert, mock_expert.capabilities)

        # Sin shadow
        engine_no_shadow = MoEPredictionEngine(registry=registry, sparsity_k=1)
        ctx = FeatureContext(
            regime="stable", mean=40.0, std=2.0, slope=0.1,
            curvature=0.0, noise_ratio=0.05, stability=0.9,
            hampel_outlier_mask=[], spatial_correlation_score=0.0,
        )
        result_no_shadow = engine_no_shadow.predict_with_context([40.0, 41.0, 42.0], None, ctx)

        # Con shadow (TreeGatingNetwork no entrenada, fallback uniforme)
        shadow_gating = TreeGatingNetwork(expert_ids=["baseline"])
        engine_with_shadow = MoEPredictionEngine(
            registry=registry, sparsity_k=1, shadow_gating=shadow_gating
        )
        result_with_shadow = engine_with_shadow.predict_with_context([40.0, 41.0, 42.0], None, ctx)

        assert result_with_shadow.predicted_value == result_no_shadow.predicted_value
        assert result_with_shadow.confidence == result_no_shadow.confidence
        # Metadata debe incluir shadow_gating
        assert "shadow_gating" in result_with_shadow.metadata


class TestDispatcherTimeout:
    """test_dispatcher_continues_on_expert_timeout — timeout no rompe el flujo."""

    def test_dispatcher_continues_on_expert_timeout(self):
        """Un experto que tarda 300ms con timeout 200ms es omitido; los otros se ejecutan."""
        import time
        registry = ExpertRegistry()

        # Experto rápido
        fast = MagicMock()
        fast.can_handle.return_value = True
        fast.predict.return_value = ExpertOutput(
            prediction=10.0, confidence=0.9, trend="stable"
        )
        fast.capabilities = ExpertCapability(regimes=("stable",), min_points=1)
        registry.register("fast", fast, fast.capabilities)

        # Experto lento (simulado con sleep)
        slow = MagicMock()
        slow.can_handle.return_value = True

        def slow_predict(window):
            time.sleep(0.3)
            return ExpertOutput(prediction=20.0, confidence=0.8, trend="stable")

        slow.predict = slow_predict
        slow.capabilities = ExpertCapability(regimes=("stable",), min_points=1)
        registry.register("slow", slow, slow.capabilities)

        from iot_machine_learning.infrastructure.ml.moe.gateway.expert_dispatcher import ExpertDispatcher
        dispatcher = ExpertDispatcher(registry, timeout_ms=200)
        from iot_machine_learning.domain.entities.iot.sensor_reading import SensorWindow, Reading
        window = SensorWindow(series_id="test", readings=[
            Reading(series_id="test", value=1.0, timestamp=0.0),
        ])

        outputs = dispatcher.dispatch(["fast", "slow"], window)

        # Fast debe estar presente; slow debe ser omitido por timeout
        assert "fast" in outputs
        assert outputs["fast"].prediction == 10.0
        assert "slow" not in outputs


class TestABLogSchema:
    """test_ab_log_schema — verifica que el log de A/B tiene todos los campos requeridos."""

    def test_ab_log_schema(self):
        from iot_machine_learning.infrastructure.ml.moe.ab.moe_ab_logger import MoEABLogger, ABLogEntry

        logger = MoEABLogger()
        entry = ABLogEntry(
            timestamp="2026-01-01T00:00:00",
            cell="B",
            engine_used="moe_engine",
            prediction_value=42.0,
            confidence=0.85,
            latency_ms=15.2,
            regime="stable",
            expert_weights={"baseline": 0.8, "taylor": 0.2},
            selected_experts=["baseline"],
            dominant_expert="baseline",
            actual_value=41.5,
            metadata={"shadow_gating": {"max_prob_diff": 0.05}},
        )
        logger.log_prediction(entry)

        report = logger.generate_report()
        assert report["status"] == "ok"
        assert report["total_predictions"] == 1
        assert report["cell_b"]["count"] == 1
        assert report["cell_a"]["count"] == 0
        assert report["cell_b"]["mae"] is not None
        assert "expert_distribution" in report["cell_b"]
        assert report["cell_b"]["expert_distribution"]["baseline"] == 1.0

    def test_ab_log_entry_to_dict_has_required_fields(self):
        from iot_machine_learning.infrastructure.ml.moe.ab.moe_ab_logger import ABLogEntry

        entry = ABLogEntry(
            timestamp="2026-01-01T00:00:00",
            cell="B",
            engine_used="moe_engine",
            prediction_value=42.0,
            confidence=0.85,
            latency_ms=15.2,
            regime="stable",
            expert_weights={"baseline": 0.8},
            selected_experts=["baseline"],
            dominant_expert="baseline",
        )
        d = entry.to_dict()
        required = [
            "timestamp", "cell", "engine_used", "prediction_value",
            "confidence", "latency_ms", "regime", "expert_weights",
            "selected_experts", "dominant_expert",
        ]
        for field in required:
            assert field in d, f"Missing field: {field}"
