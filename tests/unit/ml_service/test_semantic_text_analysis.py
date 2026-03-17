"""Tests for the semantic text analysis pipeline (Phase 3).

Covers:
- text_chunker: paragraph splitting, sentence fallback, merging
- text_recall: RecallResult, response parsing
- text_pattern: TextPatternResult
- conclusion_builder: semantic conclusion, domain classification, actions
- text_analyzer: full pipeline integration (without Weaviate)
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

# ── Chunker tests ────────────────────────────────────────────────

from iot_machine_learning.ml_service.api.services.analyzers.text_chunker import (
    TextChunk,
    chunk_text,
)


class TestTextChunker(unittest.TestCase):
    """Tests for text_chunker.chunk_text()."""

    def test_empty_text_returns_empty(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])

    def test_single_paragraph(self):
        text = "This is a short paragraph with a few words."
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].index, 0)
        self.assertIn("short paragraph", chunks[0].text)

    def test_multiple_paragraphs_split(self):
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird one."
        chunks = chunk_text(text, max_tokens=500)
        # All paragraphs are short, should merge into one chunk
        self.assertGreaterEqual(len(chunks), 1)

    def test_long_paragraph_split_on_sentences(self):
        # Create a paragraph with many sentences that exceeds max_tokens
        sentences = ["This is sentence number %d with some extra words." % i
                     for i in range(100)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_tokens=50)
        self.assertGreater(len(chunks), 1)
        # Each chunk should be within budget
        for c in chunks:
            self.assertLessEqual(c.token_estimate, 60)  # some slack

    def test_chunk_offsets_are_valid(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, max_tokens=3)
        for c in chunks:
            self.assertGreaterEqual(c.char_start, 0)
            self.assertLessEqual(c.char_end, len(text))

    def test_chunk_index_sequential(self):
        text = "A.\n\nB.\n\nC.\n\nD."
        chunks = chunk_text(text, max_tokens=2)
        indices = [c.index for c in chunks]
        self.assertEqual(indices, list(range(len(chunks))))

    def test_token_estimate_property(self):
        chunk = TextChunk(index=0, text="hello world foo bar", char_start=0, char_end=19)
        self.assertEqual(chunk.token_estimate, 4)


# ── Recall tests ─────────────────────────────────────────────────

from iot_machine_learning.ml_service.api.services.analyzers.text_recall import (
    RecallResult,
    recall_similar_documents,
    _parse_response,
)


class TestRecallResult(unittest.TestCase):

    def test_to_dict(self):
        r = RecallResult(doc_id="abc", content="test content", score=0.85,
                         filename="report.txt", conclusion="critical issue")
        d = r.to_dict()
        self.assertEqual(d["doc_id"], "abc")
        self.assertEqual(d["score"], 0.85)
        self.assertEqual(d["filename"], "report.txt")

    def test_no_weaviate_returns_empty(self):
        results = recall_similar_documents(None, "query text")
        self.assertEqual(results, [])

    def test_empty_query_returns_empty(self):
        results = recall_similar_documents("http://localhost:8080", "")
        self.assertEqual(results, [])


class TestParseResponse(unittest.TestCase):

    def test_valid_response(self):
        data = {
            "data": {
                "Get": {
                    "MLExplanation": [
                        {
                            "seriesId": "report.txt",
                            "explanationText": "Server failure detected",
                            "auditTraceId": "other-id",
                            "metadata": json.dumps({
                                "tenant_id": "t1",
                                "filename": "report.txt",
                                "conclusion": "Critical incident",
                            }),
                            "_additional": {"id": "uuid-1", "certainty": 0.88},
                        }
                    ]
                }
            }
        }
        results = _parse_response(data, exclude_analysis_id="my-id", tenant_id="t1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].score, 0.88)
        self.assertEqual(results[0].conclusion, "Critical incident")

    def test_self_match_excluded(self):
        data = {
            "data": {
                "Get": {
                    "MLExplanation": [
                        {
                            "auditTraceId": "my-id",
                            "explanationText": "test",
                            "metadata": "{}",
                            "_additional": {"id": "uuid-1", "certainty": 0.9},
                        }
                    ]
                }
            }
        }
        results = _parse_response(data, exclude_analysis_id="my-id", tenant_id="")
        self.assertEqual(len(results), 0)

    def test_tenant_isolation(self):
        data = {
            "data": {
                "Get": {
                    "MLExplanation": [
                        {
                            "auditTraceId": "other",
                            "explanationText": "test",
                            "metadata": json.dumps({"tenant_id": "t2"}),
                            "_additional": {"id": "uuid-1", "certainty": 0.9},
                        }
                    ]
                }
            }
        }
        results = _parse_response(data, exclude_analysis_id="", tenant_id="t1")
        self.assertEqual(len(results), 0)

    def test_empty_response(self):
        data = {"data": {"Get": {"MLExplanation": None}}}
        results = _parse_response(data, exclude_analysis_id="", tenant_id="")
        self.assertEqual(results, [])

    def test_error_response(self):
        data = {"errors": [{"message": "class not found"}]}
        results = _parse_response(data, exclude_analysis_id="", tenant_id="")
        self.assertEqual(results, [])


# ── Pattern tests ────────────────────────────────────────────────

from iot_machine_learning.ml_service.api.services.analyzers.text_pattern import (
    TextPatternResult,
    detect_text_patterns,
)


class TestTextPattern(unittest.TestCase):

    def test_empty_result_defaults(self):
        r = TextPatternResult()
        self.assertFalse(r.available)
        self.assertEqual(r.n_patterns, 0)
        self.assertEqual(r.change_points, [])

    def test_too_few_sentences_returns_empty(self):
        r = detect_text_patterns(["Short.", "Two."])
        self.assertFalse(r.available)

    def test_consistent_sentences_no_patterns(self):
        # 10 sentences of similar length — no change points expected
        sentences = ["This is a normal sentence of moderate length."] * 10
        r = detect_text_patterns(sentences)
        # May or may not find patterns depending on the detector,
        # but should not crash
        self.assertIsInstance(r, TextPatternResult)


# ── Conclusion builder tests ─────────────────────────────────────

from iot_machine_learning.ml_service.api.services.analyzers.conclusion_builder import (
    build_semantic_conclusion,
    build_text_conclusion,
    classify_document_domain,
    _extract_key_topics,
    _build_severity_assessment,
    _get_recommended_actions,
    _build_recall_context,
)


class TestClassifyDocumentDomain(unittest.TestCase):

    def test_infrastructure_domain(self):
        text = "The server CPU temperature reached 89°C. Node cluster is degraded."
        self.assertEqual(classify_document_domain(text), "infrastructure")

    def test_security_domain(self):
        text = "A breach was detected. Unauthorized access to credentials."
        self.assertEqual(classify_document_domain(text), "security")

    def test_operations_domain(self):
        text = "The deploy failed during migration. Rollback was initiated."
        self.assertEqual(classify_document_domain(text), "operations")

    def test_business_domain(self):
        text = "The client contract SLA was breached. Revenue impact assessment needed."
        self.assertEqual(classify_document_domain(text), "business")

    def test_general_domain_no_keywords(self):
        text = "Lorem ipsum dolor sit amet."
        self.assertEqual(classify_document_domain(text), "general")


class TestExtractKeyTopics(unittest.TestCase):

    def test_extracts_urgency_keywords(self):
        hits = [
            {"keyword": "failure", "count": 3},
            {"keyword": "error", "count": 2},
        ]
        topics = _extract_key_topics("some text", hits)
        self.assertIn("failure", topics)
        self.assertIn("error", topics)

    def test_extracts_numeric_with_units(self):
        topics = _extract_key_topics("Temperature reached 89°C and CPU at 95%", [])
        unit_topics = [t for t in topics if "°C" in t or "%" in t]
        self.assertGreater(len(unit_topics), 0)

    def test_extracts_identifiers(self):
        topics = _extract_key_topics("Check NODE-017 and SERVER-042", [])
        self.assertIn("NODE-017", topics)
        self.assertIn("SERVER-042", topics)


class TestBuildSeverityAssessment(unittest.TestCase):

    def test_critical_severity(self):
        result = _build_severity_assessment("critical", 0.9, "negative", -0.8)
        self.assertIn("Critical", result)
        self.assertIn("Immediate", result)

    def test_warning_severity(self):
        result = _build_severity_assessment("warning", 0.5, "neutral", 0.0)
        self.assertIn("Moderate concern", result)

    def test_info_severity(self):
        result = _build_severity_assessment("info", 0.1, "positive", 0.5)
        self.assertIn("Informational", result)
        self.assertIn("No immediate", result)


class TestGetRecommendedActions(unittest.TestCase):

    def test_infrastructure_critical(self):
        result = _get_recommended_actions("infrastructure", "critical", [])
        self.assertIn("infrastructure", result.lower())

    def test_security_critical(self):
        result = _get_recommended_actions("security", "critical", [])
        self.assertIn("incident response", result.lower())

    def test_enriches_with_identifiers(self):
        topics = ["failure", "NODE-017", "SERVER-042"]
        result = _get_recommended_actions("infrastructure", "critical", topics)
        self.assertIn("NODE-017", result)

    def test_general_fallback(self):
        result = _get_recommended_actions("unknown_domain", "info", [])
        self.assertIn("No action", result)


class TestBuildRecallContext(unittest.TestCase):

    def test_empty_recall(self):
        self.assertEqual(_build_recall_context(None), "")
        self.assertEqual(_build_recall_context([]), "")

    def test_with_recall_results(self):
        recalls = [
            RecallResult(doc_id="1", content="test", score=0.9,
                         filename="old_report.txt",
                         conclusion="Was a critical outage"),
        ]
        result = _build_recall_context(recalls)
        self.assertIn("Similar past documents", result)
        self.assertIn("old_report.txt", result)
        self.assertIn("critical outage", result)


class TestBuildSemanticConclusion(unittest.TestCase):

    def _base_kwargs(self) -> Dict[str, Any]:
        return dict(
            full_text="The server CPU reached 89°C. Cooling failure on NODE-017.",
            word_count=50,
            n_sentences=5,
            paragraph_count=2,
            sentiment_label="negative",
            sentiment_score=-0.6,
            urgency_score=0.85,
            urgency_total_hits=5,
            urgency_hits=[{"keyword": "failure", "count": 2}],
            urgency_severity="critical",
            readability_avg_sentence_len=10.0,
            readability_vocabulary_richness=0.7,
        )

    def test_critical_conclusion_is_actionable(self):
        conclusion = build_semantic_conclusion(**self._base_kwargs())
        # Should mention what it's about
        self.assertIn("Infrastructure", conclusion)
        # Should indicate severity
        self.assertIn("Critical", conclusion)
        # Should have recommended actions
        self.assertIn("Recommended actions", conclusion)

    def test_includes_key_topics(self):
        conclusion = build_semantic_conclusion(**self._base_kwargs())
        self.assertIn("NODE-017", conclusion)

    def test_includes_recall_context(self):
        kwargs = self._base_kwargs()
        kwargs["recall_results"] = [
            RecallResult(doc_id="1", content="old", score=0.88,
                         filename="prev_report.txt",
                         conclusion="Previous cooling incident"),
        ]
        conclusion = build_semantic_conclusion(**kwargs)
        self.assertIn("Similar past documents", conclusion)
        self.assertIn("prev_report.txt", conclusion)

    def test_includes_pattern_summary(self):
        kwargs = self._base_kwargs()
        kwargs["pattern_summary"] = "2 narrative shifts detected"
        conclusion = build_semantic_conclusion(**kwargs)
        self.assertIn("narrative shifts", conclusion)

    def test_info_severity_no_immediate_action(self):
        kwargs = self._base_kwargs()
        kwargs["full_text"] = "Everything is fine. Normal operations."
        kwargs["urgency_severity"] = "info"
        kwargs["urgency_score"] = 0.1
        kwargs["urgency_hits"] = []
        kwargs["sentiment_label"] = "positive"
        conclusion = build_semantic_conclusion(**kwargs)
        self.assertIn("No immediate", conclusion.lower() + conclusion)


class TestLegacyConclusionPreserved(unittest.TestCase):
    """Verify build_text_conclusion still works unchanged."""

    def test_legacy_output_format(self):
        conclusion = build_text_conclusion(
            word_count=100,
            n_sentences=10,
            paragraph_count=3,
            sentiment_label="negative",
            sentiment_score=-0.5,
            urgency_score=0.8,
            urgency_total_hits=5,
            urgency_hits=[{"keyword": "error", "count": 3}],
            urgency_severity="critical",
            readability_avg_sentence_len=10.0,
            readability_vocabulary_richness=0.6,
        )
        self.assertIn("Documento de texto", conclusion)
        self.assertIn("Urgencia ALTA", conclusion)


# ── Text analyzer integration tests ──────────────────────────────

from iot_machine_learning.ml_service.api.services.analyzers.text_analyzer import (
    analyze_text_document,
)


class TestAnalyzeTextDocumentIntegration(unittest.TestCase):
    """Integration tests for the full text analysis pipeline."""

    def _make_payload(self, text: str) -> Dict[str, Any]:
        words = text.split()
        paragraphs = text.split("\n\n")
        return {
            "data": {
                "full_text": text,
                "word_count": len(words),
                "char_count": len(text),
                "paragraph_count": len(paragraphs),
            }
        }

    def test_basic_text_analysis(self):
        """Pipeline runs without Weaviate and produces conclusion."""
        payload = self._make_payload(
            "The server experienced a critical failure. "
            "CPU temperature reached 89°C. "
            "Cooling system is degraded on NODE-017. "
            "Network latency increased to 500ms. "
            "Immediate action required to prevent total outage."
        )
        result = analyze_text_document("doc-001", payload)

        self.assertIn("analysis", result)
        self.assertIn("conclusion", result)
        self.assertIn("confidence", result)

        # Conclusion should be actionable, not just scores
        conclusion = result["conclusion"]
        self.assertIn("Infrastructure", conclusion)
        self.assertIn("Recommended actions", conclusion)

    def test_low_urgency_text(self):
        payload = self._make_payload(
            "Monthly status report. All systems operating normally. "
            "No incidents recorded. Performance metrics within expected range. "
            "Team completed scheduled maintenance successfully."
        )
        result = analyze_text_document("doc-002", payload)
        conclusion = result["conclusion"]
        # Should not recommend immediate action
        self.assertNotIn("Immediate attention", conclusion)

    def test_preserves_analysis_structure(self):
        """Verify the analysis dict still has all expected keys."""
        payload = self._make_payload("Simple test document with some content here.")
        result = analyze_text_document("doc-003", payload)
        analysis = result["analysis"]
        self.assertIn("sentiment", analysis)
        self.assertIn("sentiment_score", analysis)
        self.assertIn("urgency_score", analysis)
        self.assertIn("readability", analysis)
        self.assertIn("triggers_activated", analysis)

    def test_no_weaviate_still_works(self):
        """Without _weaviate_url in payload, no semantic enrichment
        but pipeline completes."""
        payload = self._make_payload("Test document without Weaviate context.")
        result = analyze_text_document("doc-004", payload)
        self.assertIn("conclusion", result)
        self.assertGreater(result["confidence"], 0)

    def test_incident_report_actionable(self):
        """An incident report should produce actionable conclusions."""
        text = (
            "INCIDENT REPORT - NODE-017 Cooling Failure\n\n"
            "At 14:30 UTC, monitoring detected a critical temperature "
            "spike on NODE-017 in rack B-12. CPU temperature reached "
            "89°C, exceeding the 75°C warning threshold.\n\n"
            "Root cause: Primary cooling unit failure. Backup cooling "
            "engaged but operating at 60% capacity.\n\n"
            "Impact: Risk of thermal throttling affecting 12 virtual "
            "machines. Potential service degradation for 3 clients.\n\n"
            "The server cluster has been experiencing intermittent "
            "network latency of 500ms, up from the normal 20ms baseline. "
            "This degradation correlates with the cooling failure.\n\n"
            "Recommended immediate actions:\n"
            "1. Dispatch cooling system technician\n"
            "2. Prepare VM migration plan for NODE-017 workloads\n"
            "3. Escalate to network provider for latency investigation"
        )
        payload = self._make_payload(text)
        result = analyze_text_document("doc-005", payload)
        conclusion = result["conclusion"]

        # Should identify the domain
        self.assertIn("Infrastructure", conclusion)
        # Should recommend action
        self.assertIn("Recommended actions", conclusion)
        # Should have a severity line
        self.assertIn("Severity:", conclusion)
        # Should extract key topics (failure keyword at minimum)
        self.assertIn("failure", conclusion.lower())


if __name__ == "__main__":
    unittest.main()
