"""Document Analysis Service — thin dispatcher.

Routes by content_type to the appropriate analyzer module.
All analysis logic lives in ``analyzers/``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from .analyzers.text_analyzer import analyze_text_document
from .analyzers.tabular_analyzer import analyze_tabular_document
from .analyzers.media_analyzer import analyze_image, analyze_audio, analyze_binary

logger = logging.getLogger(__name__)

_DISPATCH = {
    "tabular": lambda doc_id, p: analyze_tabular_document(p),
    "numeric": lambda doc_id, p: analyze_tabular_document(p),
    "text":    analyze_text_document,
    "image":   lambda doc_id, p: analyze_image(p),
    "audio":   lambda doc_id, p: analyze_audio(p),
}


class DocumentAnalyzer:
    """Universal document analyzer backed by real ML engines."""

    def analyze(
        self,
        document_id: str,
        content_type: str,
        normalized_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze document and return structured result."""
        start = time.time()

        try:
            handler = _DISPATCH.get(content_type)

            if content_type == "mixed":
                result = _analyze_mixed(document_id, normalized_payload)
            elif handler:
                result = handler(document_id, normalized_payload)
            else:
                result = analyze_binary(normalized_payload)

            return {
                "document_id": document_id,
                "content_type": content_type,
                "analysis": result["analysis"],
                "adaptive_thresholds": result.get("adaptive_thresholds", {}),
                "conclusion": result["conclusion"],
                "confidence": result.get("confidence", 0.85),
                "processing_time_ms": (time.time() - start) * 1000,
            }
        except Exception as exc:
            logger.exception(
                "[DOCUMENT-ANALYZER] Error analyzing %s: %s",
                document_id, exc,
            )
            raise


def _analyze_mixed(
    document_id: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Combine text + tabular analysis."""
    text_r = analyze_text_document(document_id, payload)
    tab_r = analyze_tabular_document(payload)

    return {
        "analysis": {
            "text": text_r["analysis"],
            "tabular": tab_r["analysis"],
            "triggers_activated": (
                text_r["analysis"].get("triggers_activated", [])
                + tab_r["analysis"].get("triggers_activated", [])
            ),
        },
        "adaptive_thresholds": {
            **text_r.get("adaptive_thresholds", {}),
            **tab_r.get("adaptive_thresholds", {}),
        },
        "conclusion": (
            "=== Análisis de Texto ===\n" + text_r["conclusion"]
            + "\n\n=== Análisis Numérico ===\n" + tab_r["conclusion"]
        ),
        "confidence": round(
            (text_r.get("confidence", 0.7) + tab_r.get("confidence", 0.7)) / 2,
            3,
        ),
    }
