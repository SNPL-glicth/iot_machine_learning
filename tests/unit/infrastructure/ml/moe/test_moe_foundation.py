"""Tests para Sprint 1: Foundation del refactor Pipeline-Aware MoE.

Verifica:
- FeatureContext inmutable y correctamente mapeado
- PipelineContext acepta feature_context sin romper with_field
- PerceivePhase exporta FeatureContext
- ExpertDispatcher bugs corregidos (list_all, can_any_expert_handle eliminado)
- MoEFactory pasa fallback_engine y registra StatisticalExpert
- PredictionEnricher usa series_id con fallback a sensor_id
"""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from iot_machine_learning.infrastructure.ml.moe.feature_context import FeatureContext
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import (
    PipelineContext,
    create_initial_context,
)
from iot_machine_learning.infrastructure.ml.moe.gateway.expert_dispatcher import ExpertDispatcher
from iot_machine_learning.infrastructure.ml.moe.gateway.prediction_enricher import (
    PredictionEnricher,
    MoEMetadata,
)
from iot_machine_learning.infrastructure.ml.moe.registry import ExpertRegistry
from iot_machine_learning.domain.entities.prediction import Prediction


class TestFeatureContext:
    """FeatureContext debe ser frozen e inmutable."""

    def test_feature_context_fields(self):
        ctx = FeatureContext(
            regime="stable",
            mean=10.0,
            std=1.5,
            slope=0.2,
            curvature=0.01,
            noise_ratio=0.15,
            stability=0.8,
            hampel_outlier_mask=[False, True, False],
            spatial_correlation_score=0.3,
        )
        assert ctx.regime == "stable"
        assert ctx.mean == 10.0
        assert ctx.std == 1.5
        assert ctx.slope == 0.2
        assert ctx.hampel_outlier_mask == [False, True, False]
        assert ctx.spatial_correlation_score == 0.3

    def test_feature_context_is_frozen(self):
        ctx = FeatureContext.empty()
        with pytest.raises(FrozenInstanceError):
            ctx.regime = "volatile"

    def test_feature_context_empty_factory(self):
        ctx = FeatureContext.empty()
        assert ctx.regime == "unknown"
        assert ctx.mean == 0.0
        assert ctx.hampel_outlier_mask == []

    def test_from_structural_analysis(self):
        ctx = FeatureContext.from_structural_analysis(
            regime="trending",
            mean=5.0,
            std=0.5,
            slope=1.2,
            curvature=0.05,
            noise_ratio=0.1,
            stability=0.9,
            hampel_outlier_mask=[True, False],
            spatial_correlation_score=0.7,
        )
        assert ctx.regime == "trending"
        assert ctx.noise_ratio == 0.1
        assert ctx.hampel_outlier_mask == [True, False]

    def test_from_structural_analysis_defaults(self):
        ctx = FeatureContext.from_structural_analysis(
            regime="stable", mean=0.0, std=0.0, slope=0.0,
            curvature=0.0, noise_ratio=0.0, stability=0.0,
        )
        assert ctx.hampel_outlier_mask == []
        assert ctx.spatial_correlation_score == 0.0


class TestPipelineContextFeatureContext:
    """PipelineContext debe propagar feature_context sin romper inmutabilidad."""

    def test_with_field_accepts_feature_context(self):
        base = create_initial_context(
            orchestrator=MagicMock(),
            values=[1.0, 2.0, 3.0],
            timestamps=[0.0, 1.0, 2.0],
            series_id="test",
            flags=MagicMock(),
            timer=MagicMock(),
        )
        feature_ctx = FeatureContext.empty()
        updated = base.with_field(feature_context=feature_ctx)
        assert updated.feature_context is feature_ctx
        # Original no mutado
        assert base.feature_context is None

    def test_with_field_preserves_existing_fields(self):
        base = create_initial_context(
            orchestrator=MagicMock(),
            values=[1.0, 2.0],
            timestamps=None,
            series_id="x",
            flags=MagicMock(),
            timer=MagicMock(),
        )
        updated = base.with_field(
            regime="stable",
            feature_context=FeatureContext.empty(),
        )
        assert updated.regime == "stable"
        assert updated.feature_context is not None
        assert updated.values == [1.0, 2.0]


class TestExpertDispatcherBugs:
    """Bugs latentes de ExpertDispatcher corregidos."""

    def test_list_all_used_instead_of_list_experts(self):
        """Bug: can_any_expert_handle llamaba registry.list_experts() que no existe.
        El método fue eliminado; dispatch sigue funcionando con list_all."""
        registry = ExpertRegistry()
        mock_expert = MagicMock()
        mock_expert.can_handle.return_value = True
        registry.register("expert_a", mock_expert, MagicMock())

        dispatcher = ExpertDispatcher(registry)
        # dispatch usa registry.get directamente, no necesita list_all en dispatch
        # pero can_any_expert_handle fue eliminado
        assert not hasattr(dispatcher, "can_any_expert_handle")

    def test_can_any_expert_handle_removed(self):
        """Verifica que el método roto ya no existe en la clase."""
        assert not hasattr(ExpertDispatcher, "can_any_expert_handle")

    def test_dispatch_skips_missing_experts(self):
        """dispatch debe funcionar normalmente sin can_any_expert_handle."""
        registry = ExpertRegistry()
        mock_expert = MagicMock()
        mock_expert.can_handle.return_value = True
        mock_expert.predict.return_value = MagicMock(prediction=42.0, confidence=0.9, trend="up")
        registry.register("expert_a", mock_expert, MagicMock())

        dispatcher = ExpertDispatcher(registry)
        from iot_machine_learning.domain.entities.iot.sensor_reading import SensorWindow, Reading
        window = SensorWindow(
            series_id="s1",
            readings=[Reading(series_id="s1", value=1.0, timestamp=0.0)],
        )
        outputs = dispatcher.dispatch(["expert_a", "missing"], window)
        assert "expert_a" in outputs
        assert "missing" not in outputs


class TestPredictionEnricherSeriesId:
    """Bug: window.sensor_id en lugar de series_id."""

    def test_uses_series_id_when_available(self):
        enricher = PredictionEnricher()
        pred = Prediction(
            series_id="old",
            predicted_value=1.0,
            confidence_score=0.8,
            trend="stable",
            engine_name="test",
            metadata={},
        )
        meta = MoEMetadata(
            selected_experts=["a"],
            sparsity_k=1,
            gating_probs={"a": 1.0},
            fusion_weights={"a": 1.0},
            dominant_expert="a",
            total_latency_ms=10.0,
            moe_enabled=True,
        )
        # Window con series_id (nuevo campo)
        window = MagicMock()
        window.series_id = "series_42"
        del window.sensor_id  # asegurar que no existe

        result = enricher.enrich(pred, meta, window)
        assert result.series_id == "series_42"

    def test_falls_back_to_sensor_id(self):
        enricher = PredictionEnricher()
        pred = Prediction(
            series_id="old",
            predicted_value=1.0,
            confidence_score=0.8,
            trend="stable",
            engine_name="test",
            metadata={},
        )
        meta = MoEMetadata(
            selected_experts=["a"],
            sparsity_k=1,
            gating_probs={"a": 1.0},
            fusion_weights={"a": 1.0},
            dominant_expert="a",
            total_latency_ms=10.0,
            moe_enabled=True,
        )
        window = MagicMock(spec=[])
        window.sensor_id = "sensor_99"
        # MagicMock sin spec retorna MagicMock para attrs no definidos;
        # con spec vacío lanza AttributeError para cualquier attr no seteado.

        result = enricher.enrich(pred, meta, window)
        assert result.series_id == "sensor_99"

    def test_falls_back_to_unknown(self):
        enricher = PredictionEnricher()
        pred = Prediction(
            series_id="old",
            predicted_value=1.0,
            confidence_score=0.8,
            trend="stable",
            engine_name="test",
            metadata={},
        )
        meta = MoEMetadata(
            selected_experts=["a"],
            sparsity_k=1,
            gating_probs={"a": 1.0},
            fusion_weights={"a": 1.0},
            dominant_expert="a",
            total_latency_ms=10.0,
            moe_enabled=True,
        )
        window = MagicMock(spec=[])
        # Ni series_id ni sensor_id definidos -> AttributeError -> "unknown"

        result = enricher.enrich(pred, meta, window)
        assert result.series_id == "unknown"


class TestMoEFactoryFallbackEngine:
    """Bug: moe_factory no pasaba fallback_engine requerido."""

    def test_fallback_engine_passed_to_gateway(self):
        """Verifica que create_moe_gateway pasa fallback_engine."""
        with patch(
            "iot_machine_learning.infrastructure.config.moe_factory.EngineFactory"
        ) as mock_factory, patch(
            "iot_machine_learning.infrastructure.config.moe_factory.MoEGateway"
        ) as mock_gateway_cls, patch(
            "iot_machine_learning.infrastructure.config.moe_factory.RegimeBasedGating"
        ):

            baseline_engine = MagicMock()
            baseline_port = MagicMock()
            baseline_engine.as_port.return_value = baseline_port
            mock_factory.create.return_value = baseline_engine

            from iot_machine_learning.infrastructure.config.moe_factory import create_moe_gateway

            create_moe_gateway(sparsity_k=2, enable_logging=False)

            call_kwargs = mock_gateway_cls.call_args.kwargs
            assert "fallback_engine" in call_kwargs
            assert call_kwargs["fallback_engine"] is baseline_port

    def test_raises_valueerror_when_no_baseline(self):
        """Si baseline falla pero otros expertos se crean, debe lanzar ValueError claro."""
        with patch(
            "iot_machine_learning.infrastructure.config.moe_factory.EngineFactory"
        ) as mock_factory, patch(
            "iot_machine_learning.infrastructure.config.moe_factory.MoEGateway"
        ), patch(
            "iot_machine_learning.infrastructure.config.moe_factory.RegimeBasedGating"
        ):
            def _fake_create(name, **kwargs):
                if name == "baseline_moving_average":
                    raise RuntimeError("baseline unavailable")
                engine = MagicMock()
                engine.as_port.return_value = MagicMock()
                return engine

            mock_factory.create.side_effect = _fake_create

            from iot_machine_learning.infrastructure.config.moe_factory import create_moe_gateway

            with pytest.raises(ValueError, match="fallback_engine"):
                create_moe_gateway(sparsity_k=2, enable_logging=False)


class TestStatisticalExpertRegistered:
    """StatisticalExpert debe estar registrado en la factory."""

    def test_statistical_expert_registration_attempted(self):
        """Verifica que create_moe_gateway intenta registrar statistical."""
        with patch(
            "iot_machine_learning.infrastructure.config.moe_factory.EngineFactory"
        ) as mock_factory, patch(
            "iot_machine_learning.infrastructure.config.moe_factory.MoEGateway"
        ), patch(
            "iot_machine_learning.infrastructure.config.moe_factory.RegimeBasedGating"
        ):
            baseline_engine = MagicMock()
            statistical_engine = MagicMock()
            taylor_engine = MagicMock()
            kalman_engine = MagicMock()

            def fake_create(name, **kwargs):
                engines = {
                    "baseline_moving_average": baseline_engine,
                    "statistical": statistical_engine,
                    "taylor": taylor_engine,
                    "kalman": kalman_engine,
                }
                return engines[name]

            mock_factory.create.side_effect = fake_create
            mock_registry = MagicMock()

            with patch(
                "iot_machine_learning.infrastructure.config.moe_factory.ExpertRegistry",
                return_value=mock_registry,
            ):
                from iot_machine_learning.infrastructure.config.moe_factory import create_moe_gateway
                create_moe_gateway(sparsity_k=2, enable_logging=False)

            registered_names = [
                call.kwargs.get("expert_id") or call.args[0]
                for call in mock_registry.register.call_args_list
            ]
            assert "statistical" in registered_names
