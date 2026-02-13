"""Unit tests for MemoryRecallEnricher domain service.

Tests verify:
    - Enrichment with similar explanations and anomalies
    - Empty results when no matches found
    - Fail-safe: exceptions are caught and empty context returned
    - Query construction from prediction metadata
    - Historical reference formatting
    - Enriched explanation text composition
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from iot_machine_learning.domain.entities.memory_search_result import (
    MemorySearchResult,
)
from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.ports.cognitive_memory_port import (
    CognitiveMemoryPort,
)
from iot_machine_learning.domain.services.memory_recall_enricher import (
    MemoryRecallContext,
    MemoryRecallEnricher,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cognitive() -> MagicMock:
    mock = MagicMock(spec=CognitiveMemoryPort)
    mock.recall_similar_explanations.return_value = []
    mock.recall_similar_anomalies.return_value = []
    return mock


@pytest.fixture
def sample_prediction() -> Prediction:
    return Prediction(
        series_id="42",
        predicted_value=25.5,
        confidence_score=0.85,
        trend="up",
        engine_name="taylor",
        horizon_steps=1,
        metadata={"explanation": "Upward trend with high confidence"},
        audit_trace_id="trace-recall-001",
    )


@pytest.fixture
def sample_explanation_results() -> list:
    return [
        MemorySearchResult(
            memory_id="uuid-1",
            series_id="42",
            text="Previous upward trend detected",
            certainty=0.91,
            source_record_id=100,
            created_at="2025-02-01T10:00:00Z",
            metadata={"engine_name": "taylor"},
        ),
        MemorySearchResult(
            memory_id="uuid-2",
            series_id="42",
            text="Gradual increase over 2h window",
            certainty=0.82,
            source_record_id=101,
            created_at="2025-01-28T14:30:00Z",
            metadata={"engine_name": "taylor"},
        ),
    ]


@pytest.fixture
def sample_anomaly_results() -> list:
    return [
        MemorySearchResult(
            memory_id="uuid-3",
            series_id="42",
            text="Sudden variance spike after stability",
            certainty=0.88,
            source_record_id=200,
            created_at="2025-01-15T08:00:00Z",
            metadata={"severity": "high"},
        ),
    ]


# ---------------------------------------------------------------------------
# Test: MemoryRecallContext
# ---------------------------------------------------------------------------

class TestMemoryRecallContext:
    def test_empty_context_has_no_context(self):
        ctx = MemoryRecallContext()
        assert not ctx.has_context

    def test_context_with_explanations_has_context(self, sample_explanation_results):
        ctx = MemoryRecallContext(similar_explanations=sample_explanation_results)
        assert ctx.has_context

    def test_context_with_anomalies_has_context(self, sample_anomaly_results):
        ctx = MemoryRecallContext(similar_anomalies=sample_anomaly_results)
        assert ctx.has_context

    def test_to_dict_empty(self):
        ctx = MemoryRecallContext()
        assert ctx.to_dict() == {}

    def test_to_dict_with_data(self, sample_explanation_results):
        ctx = MemoryRecallContext(
            similar_explanations=sample_explanation_results,
            enriched_explanation="Test enriched",
            historical_references=["ref1"],
        )
        d = ctx.to_dict()
        assert d["enriched_explanation"] == "Test enriched"
        assert d["historical_references"] == ["ref1"]
        assert len(d["similar_explanations"]) == 2


# ---------------------------------------------------------------------------
# Test: enrichment with results
# ---------------------------------------------------------------------------

class TestEnrichWithResults:
    def test_returns_context_with_explanations(
        self, mock_cognitive, sample_prediction, sample_explanation_results
    ):
        mock_cognitive.recall_similar_explanations.return_value = (
            sample_explanation_results
        )

        enricher = MemoryRecallEnricher(mock_cognitive)
        ctx = enricher.enrich(sample_prediction)

        assert ctx.has_context
        assert len(ctx.similar_explanations) == 2
        assert ctx.similar_explanations[0].certainty == 0.91

    def test_returns_context_with_anomalies(
        self, mock_cognitive, sample_prediction, sample_anomaly_results
    ):
        mock_cognitive.recall_similar_anomalies.return_value = (
            sample_anomaly_results
        )

        enricher = MemoryRecallEnricher(mock_cognitive)
        ctx = enricher.enrich(sample_prediction)

        assert ctx.has_context
        assert len(ctx.similar_anomalies) == 1

    def test_returns_context_with_both(
        self,
        mock_cognitive,
        sample_prediction,
        sample_explanation_results,
        sample_anomaly_results,
    ):
        mock_cognitive.recall_similar_explanations.return_value = (
            sample_explanation_results
        )
        mock_cognitive.recall_similar_anomalies.return_value = (
            sample_anomaly_results
        )

        enricher = MemoryRecallEnricher(mock_cognitive)
        ctx = enricher.enrich(sample_prediction)

        assert ctx.has_context
        assert len(ctx.similar_explanations) == 2
        assert len(ctx.similar_anomalies) == 1
        assert len(ctx.historical_references) == 3

    def test_enriched_explanation_includes_original(
        self, mock_cognitive, sample_prediction, sample_explanation_results
    ):
        mock_cognitive.recall_similar_explanations.return_value = (
            sample_explanation_results
        )

        enricher = MemoryRecallEnricher(mock_cognitive)
        ctx = enricher.enrich(sample_prediction)

        assert "Upward trend with high confidence" in ctx.enriched_explanation
        assert "Historical context:" in ctx.enriched_explanation

    def test_historical_references_format(
        self, mock_cognitive, sample_prediction, sample_explanation_results
    ):
        mock_cognitive.recall_similar_explanations.return_value = (
            sample_explanation_results
        )

        enricher = MemoryRecallEnricher(mock_cognitive)
        ctx = enricher.enrich(sample_prediction)

        assert any(
            "2025-02-01" in ref and "certainty=0.91" in ref
            for ref in ctx.historical_references
        )


# ---------------------------------------------------------------------------
# Test: no results
# ---------------------------------------------------------------------------

class TestEnrichNoResults:
    def test_returns_empty_context_when_no_matches(
        self, mock_cognitive, sample_prediction
    ):
        enricher = MemoryRecallEnricher(mock_cognitive)
        ctx = enricher.enrich(sample_prediction)

        assert not ctx.has_context
        assert ctx.enriched_explanation == ""
        assert ctx.historical_references == []

    def test_calls_both_recall_methods(
        self, mock_cognitive, sample_prediction
    ):
        enricher = MemoryRecallEnricher(mock_cognitive)
        enricher.enrich(sample_prediction)

        mock_cognitive.recall_similar_explanations.assert_called_once()
        mock_cognitive.recall_similar_anomalies.assert_called_once()

    def test_passes_series_id_to_recall(
        self, mock_cognitive, sample_prediction
    ):
        enricher = MemoryRecallEnricher(mock_cognitive)
        enricher.enrich(sample_prediction)

        call_kwargs = mock_cognitive.recall_similar_explanations.call_args[1]
        assert call_kwargs["series_id"] == "42"
        assert call_kwargs["engine_name"] == "taylor"

    def test_passes_top_k_and_min_certainty(
        self, mock_cognitive, sample_prediction
    ):
        enricher = MemoryRecallEnricher(
            mock_cognitive, top_k=5, min_certainty=0.8
        )
        enricher.enrich(sample_prediction)

        call_kwargs = mock_cognitive.recall_similar_explanations.call_args[1]
        assert call_kwargs["limit"] == 5
        assert call_kwargs["min_certainty"] == 0.8


# ---------------------------------------------------------------------------
# Test: fail-safe
# ---------------------------------------------------------------------------

class TestEnrichFailSafe:
    def test_returns_empty_on_explanation_recall_error(
        self, mock_cognitive, sample_prediction
    ):
        mock_cognitive.recall_similar_explanations.side_effect = Exception(
            "Weaviate down"
        )

        enricher = MemoryRecallEnricher(mock_cognitive)
        ctx = enricher.enrich(sample_prediction)

        assert not ctx.has_context

    def test_returns_empty_on_anomaly_recall_error(
        self, mock_cognitive, sample_prediction
    ):
        mock_cognitive.recall_similar_anomalies.side_effect = Exception(
            "Weaviate timeout"
        )

        enricher = MemoryRecallEnricher(mock_cognitive)
        ctx = enricher.enrich(sample_prediction)

        assert not ctx.has_context

    def test_never_raises(self, mock_cognitive, sample_prediction):
        mock_cognitive.recall_similar_explanations.side_effect = RuntimeError(
            "crash"
        )
        mock_cognitive.recall_similar_anomalies.side_effect = RuntimeError(
            "crash"
        )

        enricher = MemoryRecallEnricher(mock_cognitive)
        # Should not raise
        ctx = enricher.enrich(sample_prediction)
        assert not ctx.has_context


# ---------------------------------------------------------------------------
# Test: prediction without explanation metadata
# ---------------------------------------------------------------------------

class TestEnrichNoExplanation:
    def test_still_builds_query_from_series_and_trend(
        self, mock_cognitive
    ):
        prediction = Prediction(
            series_id="99",
            predicted_value=10.0,
            confidence_score=0.5,
            trend="stable",
            engine_name="baseline",
        )

        enricher = MemoryRecallEnricher(mock_cognitive)
        enricher.enrich(prediction)

        query = mock_cognitive.recall_similar_explanations.call_args[0][0]
        assert "series 99" in query
        assert "trend stable" in query
        assert "confidence 0.50" in query
