"""Legacy document analysis pipeline.

Fallback analyzers for when universal engines are unavailable.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Legacy fallback imports
try:
    from ..analyzers.text_analyzer import analyze_text_document
    from ..analyzers.tabular_analyzer import analyze_tabular_document
    from ..analyzers.media_analyzer import analyze_image, analyze_audio, analyze_binary
    _LEGACY_AVAILABLE = True
except Exception:
    _LEGACY_AVAILABLE = False


def analyze_with_legacy(
    document_id: str,
    content_type: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Fallback to legacy analyzers.
    
    Args:
        document_id: Document identifier
        content_type: Content type (text, tabular, mixed, etc.)
        payload: Normalized payload
        
    Returns:
        Legacy analysis result
        
    Raises:
        RuntimeError: If legacy analyzers unavailable
    """
    if not _LEGACY_AVAILABLE:
        raise RuntimeError("Legacy analyzers unavailable")
    
    logger.info(f"using_legacy_analyzer_for_{content_type}")
    
    if content_type == "mixed":
        return _analyze_mixed_legacy(document_id, payload)
    elif content_type in ("text",):
        return analyze_text_document(document_id, payload)
    elif content_type in ("tabular", "numeric"):
        return analyze_tabular_document(payload)
    elif content_type == "image":
        return analyze_image(payload)
    elif content_type == "audio":
        return analyze_audio(payload)
    else:
        return analyze_binary(payload)


def _analyze_mixed_legacy(
    document_id: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Legacy mixed analysis.
    
    Args:
        document_id: Document identifier
        payload: Normalized payload
        
    Returns:
        Combined text + tabular analysis
    """
    try:
        text_r = analyze_text_document(document_id, payload)
        tab_r = analyze_tabular_document(payload)

        return {
            "analysis": {
                "text": text_r["analysis"],
                "tabular": tab_r["analysis"],
            },
            "adaptive_thresholds": {},
            "conclusion": (
                "=== Text Analysis ===\n" + text_r["conclusion"]
                + "\n\n=== Numeric Analysis ===\n" + tab_r["conclusion"]
            ),
            "confidence": round(
                (text_r.get("confidence", 0.7) + tab_r.get("confidence", 0.7)) / 2,
                3,
            ),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Legacy pipeline failed: {str(e)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
