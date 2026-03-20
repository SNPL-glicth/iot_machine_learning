"""Tests for semantic name generator."""

from __future__ import annotations

import pytest
from datetime import datetime

from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.semantic_namer import (
    generate_semantic_name,
    truncate_semantic_name,
    _extract_meaningful_words,
    _looks_like_id,
)


class TestSemanticNameGeneration:
    """Test semantic name generation."""

    def test_basic_generation(self) -> None:
        """Should generate semantic name from conclusion."""
        conclusion = "Incidente crítico en infraestructura TMP-004 enfriamiento"
        domain = "infrastructure"
        date = datetime(2024, 3, 20)
        
        name = generate_semantic_name(conclusion, domain, date)
        
        assert "Incidente" in name
        assert "Crítico" in name
        assert "TMP-004" in name or "TMP-004" in name.upper()
        assert "Mar-20" in name
        assert " — " in name

    def test_multiple_sentences(self) -> None:
        """Should extract keywords from multiple sentences."""
        conclusion = ("Sistema de enfriamiento presenta fallo crítico. "
                     "Temperatura servidor SRV-01 alcanzó 95°C. "
                     "Requiere acción inmediata.")
        domain = "infrastructure"
        date = datetime(2024, 2, 15)
        
        name = generate_semantic_name(conclusion, domain, date)
        
        # Should extract important keywords
        assert any(word in name for word in ["Sistema", "Enfriamiento", "Fallo", "Crítico", "SRV-01"])
        assert "Feb-15" in name

    def test_english_text(self) -> None:
        """Should work with English text."""
        conclusion = "Critical system failure detected on database server DB-001"
        domain = "infrastructure"
        date = datetime(2024, 1, 10)
        
        name = generate_semantic_name(conclusion, domain, date)
        
        assert "Critical" in name or "System" in name or "Failure" in name
        assert "DB-001" in name or "DB-001" in name.upper()
        assert "Jan-10" in name

    def test_mixed_language(self) -> None:
        """Should handle mixed Spanish/English."""
        conclusion = "Server critical alert: temperatura alta en TMP-999"
        domain = "infrastructure"
        date = datetime(2024, 12, 25)
        
        name = generate_semantic_name(conclusion, domain, date)
        
        assert "Dec-25" in name
        # Should extract meaningful words from both languages
        assert len(name) > 10

    def test_preserves_ids(self) -> None:
        """Should preserve identifier patterns."""
        conclusion = "Alerta en sensor TMP-004 y dispositivo DEV-123"
        domain = "infrastructure"
        date = datetime(2024, 6, 1)
        
        name = generate_semantic_name(conclusion, domain, date)
        
        # IDs should be preserved (case may vary)
        name_upper = name.upper()
        assert "TMP-004" in name_upper or "TMP-004" in name
        assert "Jun-01" in name

    def test_max_five_words(self) -> None:
        """Should limit to 5 words + date."""
        conclusion = ("Este es un texto muy largo con muchas palabras "
                     "importantes crítico urgente alerta sistema fallo "
                     "infraestructura servidor base datos")
        domain = "infrastructure"
        date = datetime(2024, 7, 4)
        
        name = generate_semantic_name(conclusion, domain, date)
        
        # Split by spaces, subtract the "—" and date
        parts = name.split(" — ")[0].split()
        assert len(parts) <= 5


class TestSemanticNameEdgeCases:
    """Test edge cases."""

    def test_empty_conclusion(self) -> None:
        """Empty conclusion should return default name."""
        name = generate_semantic_name("", "infrastructure", datetime(2024, 3, 20))
        
        assert "Análisis" in name or "Infrastructure" in name
        assert "Mar-20" in name

    def test_none_conclusion(self) -> None:
        """None conclusion should return default name."""
        name = generate_semantic_name(None, "security", datetime(2024, 3, 20))
        
        assert "Análisis" in name or "Security" in name
        assert "Mar-20" in name

    def test_only_stopwords(self) -> None:
        """Conclusion with only stopwords should return default."""
        conclusion = "el la los las de en con para"
        name = generate_semantic_name(conclusion, "general", datetime(2024, 3, 20))
        
        assert "Análisis" in name or "General" in name

    def test_very_long_name_truncated(self) -> None:
        """Very long name should be truncated to 200 chars."""
        # Create a conclusion with very long words
        conclusion = " ".join(["palabra" + str(i) * 20 for i in range(10)])
        name = generate_semantic_name(conclusion, "test", datetime(2024, 3, 20))
        
        assert len(name) <= 200

    def test_special_characters_handled(self) -> None:
        """Special characters should be handled gracefully."""
        conclusion = "Sistema @ crítico #alert $urgente %importante"
        name = generate_semantic_name(conclusion, "test", datetime(2024, 3, 20))
        
        # Should extract words, ignore special chars
        assert "Sistema" in name or "Crítico" in name


class TestMeaningfulWordExtraction:
    """Test _extract_meaningful_words helper."""

    def test_filters_stopwords(self) -> None:
        """Should filter out stopwords."""
        words = _extract_meaningful_words("el servidor está con un problema crítico")
        
        # Should keep meaningful words
        assert "servidor" in words
        assert "problema" in words
        assert "crítico" in words
        
        # Should filter stopwords
        assert "el" not in words
        assert "está" not in words
        assert "con" not in words

    def test_preserves_ids(self) -> None:
        """Should preserve ID patterns."""
        words = _extract_meaningful_words("Sensor TMP-004 y servidor SRV01 fallan")
        
        assert "tmp-004" in words or "TMP-004" in words
        assert "srv01" in words or "SRV01" in words

    def test_orders_by_importance(self) -> None:
        """Should order words by importance."""
        words = _extract_meaningful_words(
            "Critical critical CRITICAL importante TMP-001"
        )
        
        # ID patterns and capitalized/repeated words should rank higher
        assert words[0] in ("tmp-001", "critical", "TMP-001", "Critical", "CRITICAL")


class TestIdPatternRecognition:
    """Test _looks_like_id helper."""

    def test_recognizes_hyphen_ids(self) -> None:
        """Should recognize hyphen-separated IDs."""
        assert _looks_like_id("tmp-004")
        assert _looks_like_id("SRV-01")
        assert _looks_like_id("DB-999")

    def test_recognizes_no_separator_ids(self) -> None:
        """Should recognize IDs without separator."""
        assert _looks_like_id("srv01")
        assert _looks_like_id("TMP004")

    def test_rejects_non_ids(self) -> None:
        """Should reject non-ID patterns."""
        assert not _looks_like_id("servidor")
        assert not _looks_like_id("critical")
        assert not _looks_like_id("123")  # Pure number


class TestSemanticNameTruncation:
    """Test truncation utility."""

    def test_no_truncation_needed(self) -> None:
        """Short names should not be truncated."""
        name = "Test Name — Mar-20"
        truncated = truncate_semantic_name(name, max_length=200)
        
        assert truncated == name

    def test_truncates_at_word_boundary(self) -> None:
        """Should truncate at word boundary."""
        name = "Very Long Name With Many Words That Exceeds Maximum Length — Mar-20"
        truncated = truncate_semantic_name(name, max_length=30)
        
        assert len(truncated) <= 30
        assert truncated.endswith("...")

    def test_custom_max_length(self) -> None:
        """Should respect custom max_length."""
        name = "Test Name — Mar-20"
        truncated = truncate_semantic_name(name, max_length=10)
        
        assert len(truncated) <= 10
