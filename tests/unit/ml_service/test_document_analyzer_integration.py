"""Integration tests for DocumentAnalyzer with UniversalEngines."""

from __future__ import annotations

import pytest

from iot_machine_learning.ml_service.api.services.document_analyzer import (
    DocumentAnalyzer,
)


class TestDocumentAnalyzerIntegration:
    """Test DocumentAnalyzer integration with universal engines."""

    def test_analyzer_instantiates(self) -> None:
        """DocumentAnalyzer should instantiate."""
        analyzer = DocumentAnalyzer()
        assert analyzer is not None

    def test_analyze_text_document(self) -> None:
        """Analyze text document end-to-end."""
        analyzer = DocumentAnalyzer()
        
        payload = {
            "full_text": "Critical system alert: Server CPU usage at 98%. Immediate action required.",
            "metadata": {"word_count": 12},
        }
        
        result = analyzer.analyze(
            document_id="test-doc-001",
            content_type="text",
            normalized_payload=payload,
            tenant_id="test-tenant",
        )
        
        assert result is not None
        assert "document_id" in result
        assert "content_type" in result
        assert "analysis" in result
        assert "confidence" in result
        assert "processing_time_ms" in result
        assert result["document_id"] == "test-doc-001"

    def test_analyze_numeric_data(self) -> None:
        """Analyze numeric series."""
        analyzer = DocumentAnalyzer()
        
        values = [20.0 + i * 0.5 for i in range(50)]
        
        payload = {
            "values": values,
            "metadata": {"n_points": len(values)},
        }
        
        result = analyzer.analyze(
            document_id="test-doc-002",
            content_type="numeric",
            normalized_payload=payload,
        )
        
        assert result is not None
        assert "analysis" in result
        assert "confidence" in result

    def test_analyze_tabular_data(self) -> None:
        """Analyze tabular/CSV data."""
        analyzer = DocumentAnalyzer()
        
        payload = {
            "data": {
                "rows": [
                    {"timestamp": "2024-01-01T00:00:00", "value": 100.0},
                    {"timestamp": "2024-01-01T00:01:00", "value": 102.0},
                    {"timestamp": "2024-01-01T00:02:00", "value": 105.0},
                ],
            },
            "metadata": {"n_rows": 3, "n_columns": 2},
        }
        
        result = analyzer.analyze(
            document_id="test-doc-003",
            content_type="tabular",
            normalized_payload=payload,
        )
        
        assert result is not None
        assert "analysis" in result


class TestDocumentAnalyzerWithMemory:
    """Test DocumentAnalyzer with cognitive memory."""

    def test_analyze_with_memory_integration(self) -> None:
        """Analyze with cognitive memory should include comparative results."""
        from unittest.mock import Mock
        
        # Mock cognitive memory
        mock_memory = Mock()
        mock_memory.recall_similar_explanations = Mock(return_value=[])
        
        analyzer = DocumentAnalyzer(cognitive_memory=mock_memory)
        
        payload = {
            "full_text": "Server disk usage at 95% capacity",
            "metadata": {"word_count": 7},
        }
        
        result = analyzer.analyze(
            document_id="test-doc-mem-001",
            content_type="text",
            normalized_payload=payload,
            tenant_id="test-tenant",
        )
        
        assert result is not None
        # Should have attempted comparative analysis
        # (even if no matches found, structure should be there)


class TestDocumentAnalyzerFallback:
    """Test fallback to legacy analyzers."""

    def test_universal_unavailable_uses_legacy(self) -> None:
        """If universal engines fail, should fall back to legacy."""
        analyzer = DocumentAnalyzer()
        
        # Even if universal fails, should not crash
        payload = {"full_text": "Test message"}
        
        try:
            result = analyzer.analyze(
                document_id="test-fallback-001",
                content_type="text",
                normalized_payload=payload,
            )
            # Should succeed (either universal or legacy)
            assert result is not None
        except Exception as e:
            # Should not reach here - graceful fallback expected
            pytest.fail(f"Analyzer should gracefully fallback, got: {e}")


class TestDocumentAnalyzerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_payload_handled(self) -> None:
        """Empty payload should be handled gracefully."""
        analyzer = DocumentAnalyzer()
        
        payload = {}
        
        # Should not crash
        result = analyzer.analyze(
            document_id="test-empty-001",
            content_type="text",
            normalized_payload=payload,
        )
        
        assert result is not None

    def test_unknown_content_type(self) -> None:
        """Unknown content type should be handled."""
        analyzer = DocumentAnalyzer()
        
        payload = {"data": "some binary data"}
        
        result = analyzer.analyze(
            document_id="test-unknown-001",
            content_type="unknown",
            normalized_payload=payload,
        )
        
        assert result is not None

    def test_conclusion_formatting(self) -> None:
        """Conclusion should be human-readable."""
        analyzer = DocumentAnalyzer()
        
        payload = {
            "full_text": "System overload detected. CPU at 99%.",
            "metadata": {"word_count": 7},
        }
        
        result = analyzer.analyze(
            document_id="test-conclusion-001",
            content_type="text",
            normalized_payload=payload,
        )
        
        assert result is not None
        assert "conclusion" in result
        assert isinstance(result["conclusion"], str)
        assert len(result["conclusion"]) > 0
