"""Domain classification and semantic conclusion helpers.

Pure text processing functions for domain detection, topic extraction,
and action recommendation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from iot_machine_learning.infrastructure.ml.cognitive.text.analyzers.keyword_config import (
    DOMAIN_KEYWORDS,
    ACTION_TEMPLATES,
)

_SEVERITY_LABELS = {
    "critical": "Critical",
    "warning": "Moderate concern",
    "info": "Informational",
}


def classify_document_domain(text: str) -> str:
    """Classify text into a domain category based on keyword density.

    Returns the domain key with the most keyword matches, or
    ``"general"`` if no clear match.
    """
    text_lower = text.lower()
    scores: Dict[str, int] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[domain] = score

    if not scores:
        return "general"

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best


def extract_key_topics(
    text: str, urgency_hits: List[Dict[str, Any]]
) -> List[str]:
    """Extract key topics: urgency keywords + numeric values mentioned."""
    topics: List[str] = []

    # Extract identifiers (NODE-xxx, SERVER-xxx, etc.) - PRIORITY
    identifiers = re.findall(
        r'\b([A-Z]{2,}[-_]\d{2,})\b', text,
    )
    # Deduplicate identifiers while preserving order
    seen = set()
    unique_identifiers = []
    for ident in identifiers:
        if ident not in seen:
            seen.add(ident)
            unique_identifiers.append(ident)
    for ident in unique_identifiers[:3]:
        topics.append(ident)

    # Extract notable numeric values with context
    # Pattern: number followed/preceded by a unit or identifier
    numeric_contexts = re.findall(
        r'(\b\d+[.,]?\d*\s*(?:°C|°F|%|GB|MB|TB|ms|sec|hrs?|min|rpm|PSI|bar|V|A|W|kW)\b)',
        text, re.IGNORECASE,
    )
    # Clean up any spaces after commas in numbers
    cleaned_contexts = [re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', nc.strip()) for nc in numeric_contexts[:3]]
    for nc in cleaned_contexts:
        topics.append(nc)

    # Top urgency keywords (sorted by count) - LOWER PRIORITY
    sorted_hits = sorted(
        urgency_hits, key=lambda h: h.get("count", 0), reverse=True
    )
    for h in sorted_hits[:3]:  # Reduced to make room for entities
        topics.append(h["keyword"])

    return topics


def build_severity_assessment(
    urgency_severity: str,
    urgency_score: float,
    sentiment_label: str,
    sentiment_score: float,
) -> str:
    """Build severity assessment line."""
    severity_text = _SEVERITY_LABELS.get(urgency_severity, "Unknown")

    if urgency_severity == "critical":
        return (
            f"Severity: {severity_text}. "
            f"High urgency detected (score {urgency_score:.2f}) "
            f"with {sentiment_label} sentiment. "
            f"Immediate attention required."
        )
    elif urgency_severity == "warning":
        return (
            f"Severity: {severity_text}. "
            f"Elevated urgency (score {urgency_score:.2f}) "
            f"with {sentiment_label} sentiment. "
            f"Review recommended within 24 hours."
        )
    else:
        return (
            f"Severity: {severity_text}. "
            f"Low urgency (score {urgency_score:.2f}), "
            f"{sentiment_label} sentiment. "
            f"No immediate action required."
        )


def build_topic_summary(
    domain_label: str,
    key_topics: List[str],
    word_count: int,
) -> str:
    """Build the 'what is this about' line."""
    if key_topics:
        topic_str = ", ".join(key_topics[:5])
        return (
            f"{domain_label} document ({word_count} words). "
            f"Key topics: {topic_str}."
        )
    return f"{domain_label} document ({word_count} words)."


def get_recommended_actions(
    domain: str, severity: str, key_topics: List[str]
) -> str:
    """Get recommended actions based on domain + severity."""
    templates = ACTION_TEMPLATES.get(domain, ACTION_TEMPLATES["general"])
    base_action = templates.get(severity, templates["info"])

    # Enrich with specific identifiers if found
    identifiers = [t for t in key_topics if re.match(r'^[A-Z]{2,}[-_]\d', t)]
    if identifiers and severity in ("critical", "warning"):
        ids_str = ", ".join(identifiers[:3])
        return f"{base_action} Specifically check: {ids_str}."

    return base_action


def build_recall_context(
    recall_results: Optional[List[Any]],
) -> str:
    """Build context line from semantic recall results."""
    if not recall_results:
        return ""

    parts: List[str] = ["Similar past documents found:"]
    for r in recall_results[:3]:
        # RecallResult has .filename, .conclusion, .score
        fname = getattr(r, "filename", "") or "unknown"
        conclusion = getattr(r, "conclusion", "") or ""
        score = getattr(r, "score", 0.0)

        if conclusion:
            parts.append(
                f"  - {fname} (similarity {score:.0%}): {conclusion[:200]}"
            )
        else:
            content = getattr(r, "content", "")[:100]
            parts.append(
                f"  - {fname} (similarity {score:.0%}): {content}..."
            )

    return "\n".join(parts)
