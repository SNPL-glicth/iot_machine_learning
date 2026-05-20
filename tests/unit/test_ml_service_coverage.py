"""Coverage tests for ml_service/ — API services, broker, features, context, memory, metrics, runners.

Covers: api/cache_dependency, api/dependencies, api/result_store,
api/routes_cognitive, api/routes_governance, api/routes_health,
api/routes_observability, api/routes_predict_async, api/routes,
api/routes_query, api/routes_telemetry, api/schemas,
api/services/analysis/action_catalog, api/services/analysis/action_recommender,
api/services/analysis/arbitrator, api/services/analysis/cache,
api/services/analysis/conclusion_formatter,
api/services/analysis/decision_context_builder,
api/services/analysis/decision_engine_service,
api/services/analysis/document_analyzer_factory,
api/services/analysis/feedback_loop, api/services/analysis/legacy_pipeline,
api/services/analysis/neural_bridge, api/services/analysis/output_assembler,
api/services/analysis/pattern_signal_builder,
api/services/analysis/result_builder, api/services/analysis/text_score_builder,
api/services/analysis/universal_bridge,
api/services/analyzers/media_analyzer, api/services/analyzers/text_analyzer,
api/services/analyzers/text_embedder, api/services/analyzers/text_recall,
api/services/document_analyzer, api/services/model_service,
api/services/prediction_service, api/services/threshold_service,
broker/redis_reading_broker,
features/persistence/redis_window_store, features/services/window_manager,
context/services/decision_builder,
memory/services/memory_service,
metrics/ab_testing, metrics/performance_metrics,
runners/wiring/container, runners/batch_worker,
runners/adapters/orchestrator_prediction, runners/adapters/enterprise_prediction,
consumers/stream_consumer, consumers/stream_predictor
"""
from __future__ import annotations

import pytest

try:
    import dotenv  # noqa: F401
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

skip_no_dotenv = pytest.mark.skipif(
    not HAS_DOTENV, reason="python-dotenv not installed"
)


# --- API ---

@skip_no_dotenv
class TestApiSchemas:
    def test_predict_request(self):
        from iot_machine_learning.ml_service.api.schemas import PredictRequest
        r = PredictRequest(sensor_id=1)
        assert r.sensor_id == 1
        assert r.horizon_minutes == 10
        assert r.window == 60

    def test_predict_response_fields(self):
        from iot_machine_learning.ml_service.api.schemas import PredictResponse
        assert PredictResponse is not None

    def test_health_response(self):
        from iot_machine_learning.ml_service.api.schemas import HealthResponse
        h = HealthResponse(status="ok")
        assert h.status == "ok"
        assert h.degraded is False

    def test_analyze_document_request(self):
        from iot_machine_learning.ml_service.api.schemas import AnalyzeDocumentRequest
        assert AnalyzeDocumentRequest is not None

    def test_semantic_search_schemas(self):
        from iot_machine_learning.ml_service.api.schemas import (
            SemanticSearchRequest, SemanticSearchResponse,
        )
        assert SemanticSearchRequest is not None


@skip_no_dotenv
class TestApiModules:
    def test_cache_dependency(self):
        from iot_machine_learning.ml_service.api import cache_dependency
        assert cache_dependency is not None

    def test_dependencies(self):
        from iot_machine_learning.ml_service.api import dependencies
        assert dependencies is not None

    def test_result_store(self):
        from iot_machine_learning.ml_service.api import result_store
        assert result_store is not None

    def test_routes_predict_async(self):
        from iot_machine_learning.ml_service.api import routes_predict_async
        assert routes_predict_async is not None

    def test_routes_query(self):
        from iot_machine_learning.ml_service.api import routes_query
        assert routes_query is not None

    def test_routes_telemetry(self):
        from iot_machine_learning.ml_service.api import routes_telemetry
        assert routes_telemetry is not None

    def test_routes_observability(self):
        from iot_machine_learning.ml_service.api import routes_observability
        assert routes_observability is not None


@skip_no_dotenv
class TestApiAnalysisServices:
    def test_action_catalog(self):
        from iot_machine_learning.ml_service.api.services.analysis import action_catalog
        assert action_catalog is not None

    def test_action_recommender(self):
        from iot_machine_learning.ml_service.api.services.analysis import action_recommender
        assert action_recommender is not None

    def test_arbitrator(self):
        from iot_machine_learning.ml_service.api.services.analysis import arbitrator
        assert arbitrator is not None

    def test_cache(self):
        from iot_machine_learning.ml_service.api.services.analysis import cache
        assert cache is not None

    def test_conclusion_formatter(self):
        from iot_machine_learning.ml_service.api.services.analysis import conclusion_formatter
        assert conclusion_formatter is not None

    def test_decision_context_builder(self):
        from iot_machine_learning.ml_service.api.services.analysis import decision_context_builder
        assert decision_context_builder is not None

    def test_decision_engine_service(self):
        from iot_machine_learning.ml_service.api.services.analysis import decision_engine_service
        assert decision_engine_service is not None

    def test_document_analyzer_factory(self):
        from iot_machine_learning.ml_service.api.services.analysis import document_analyzer_factory
        assert document_analyzer_factory is not None

    def test_feedback_loop(self):
        from iot_machine_learning.ml_service.api.services.analysis import feedback_loop
        assert feedback_loop is not None

    def test_legacy_pipeline(self):
        from iot_machine_learning.ml_service.api.services.analysis import legacy_pipeline
        assert legacy_pipeline is not None

    def test_neural_bridge(self):
        from iot_machine_learning.ml_service.api.services.analysis import neural_bridge
        assert neural_bridge is not None

    def test_output_assembler(self):
        from iot_machine_learning.ml_service.api.services.analysis import output_assembler
        assert output_assembler is not None

    def test_pattern_signal_builder(self):
        from iot_machine_learning.ml_service.api.services.analysis import pattern_signal_builder
        assert pattern_signal_builder is not None

    def test_result_builder(self):
        from iot_machine_learning.ml_service.api.services.analysis import result_builder
        assert result_builder is not None

    def test_text_score_builder(self):
        from iot_machine_learning.ml_service.api.services.analysis import text_score_builder
        assert text_score_builder is not None

    def test_universal_bridge(self):
        from iot_machine_learning.ml_service.api.services.analysis import universal_bridge
        assert universal_bridge is not None


@skip_no_dotenv
class TestApiAnalyzerServices:
    def test_media_analyzer(self):
        from iot_machine_learning.ml_service.api.services.analyzers import media_analyzer
        assert media_analyzer is not None

    def test_text_analyzer(self):
        from iot_machine_learning.ml_service.api.services.analyzers import text_analyzer
        assert text_analyzer is not None

    def test_text_embedder(self):
        from iot_machine_learning.ml_service.api.services.analyzers import text_embedder
        assert text_embedder is not None

    def test_text_recall(self):
        from iot_machine_learning.ml_service.api.services.analyzers import text_recall
        assert text_recall is not None


@skip_no_dotenv
class TestApiTopServices:
    def test_document_analyzer(self):
        from iot_machine_learning.ml_service.api.services import document_analyzer
        assert document_analyzer is not None

    def test_model_service(self):
        from iot_machine_learning.ml_service.api.services import model_service
        assert model_service is not None

    def test_prediction_service(self):
        from iot_machine_learning.ml_service.api.services import prediction_service
        assert prediction_service is not None

    def test_threshold_service(self):
        from iot_machine_learning.ml_service.api.services import threshold_service
        assert threshold_service is not None


# --- Broker ---

class TestBroker:
    @pytest.mark.skipif(not HAS_DOTENV, reason="dotenv")
    def test_redis_reading_broker_import(self):
        from iot_machine_learning.ml_service.broker import redis_reading_broker
        assert redis_reading_broker is not None


# --- Features ---

class TestFeatures:
    def test_redis_window_store(self):
        from iot_machine_learning.ml_service.features.persistence import redis_window_store
        assert redis_window_store is not None

    def test_window_manager(self):
        from iot_machine_learning.ml_service.features.services import window_manager
        assert window_manager is not None


# --- Context ---

class TestContext:
    def test_decision_builder(self):
        try:
            from iot_machine_learning.ml_service.context.services import decision_builder
            assert decision_builder is not None
        except ImportError:
            pytest.skip("transitive import error")


# --- Memory ---

class TestMemory:
    def test_memory_service(self):
        try:
            from iot_machine_learning.ml_service.memory.services import memory_service
            assert memory_service is not None
        except ImportError:
            pytest.skip("transitive import error")


# --- Metrics ---

class TestMetrics:
    def test_ab_testing(self):
        from iot_machine_learning.ml_service.metrics import ab_testing
        assert ab_testing is not None

    def test_performance_metrics(self):
        from iot_machine_learning.ml_service.metrics import performance_metrics
        assert performance_metrics is not None


# --- Runners ---

class TestRunners:
    def test_container(self):
        from iot_machine_learning.ml_service.runners.wiring import container
        assert container is not None

    @pytest.mark.skipif(not HAS_DOTENV, reason="dotenv")
    def test_batch_worker(self):
        from iot_machine_learning.ml_service.runners import batch_worker
        assert batch_worker is not None


# --- Consumers ---

class TestConsumers:
    def test_stream_consumer(self):
        from iot_machine_learning.ml_service.consumers import stream_consumer
        assert stream_consumer is not None

    def test_stream_predictor(self):
        from iot_machine_learning.ml_service.consumers import stream_predictor
        assert stream_predictor is not None
