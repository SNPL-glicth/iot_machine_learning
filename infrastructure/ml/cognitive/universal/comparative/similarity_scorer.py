"""Compute similarity metrics between current and historical analyses."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set


def compute_similarity_metrics(
    current_analysis: Dict[str, Any],
    historical_matches: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute delta metrics vs historical average.

    Args:
        current_analysis: Analysis dict from UniversalResult
        historical_matches: List of past analysis dicts from memory

    Returns:
        Dict with:
            - severity_delta_pct: % change in severity
            - urgency_delta_pct: % change in urgency
            - topic_overlap_pct: % of common topics/keywords
    """
    if not historical_matches:
        return {
            "severity_delta_pct": 0.0,
            "urgency_delta_pct": 0.0,
            "topic_overlap_pct": 0.0,
        }
    
    current_severity = _extract_severity_from_analysis(current_analysis)
    current_severity_score = _severity_to_score(current_severity)
    
    hist_severity_scores = [
        _severity_to_score(h.get("severity", "info"))
        for h in historical_matches
    ]
    avg_hist_severity = sum(hist_severity_scores) / len(hist_severity_scores)
    
    severity_delta_pct = (
        ((current_severity_score - avg_hist_severity) / max(avg_hist_severity, 0.1)) * 100
        if avg_hist_severity > 0 else 0.0
    )
    
    current_urgency = current_analysis.get("urgency_score", 0.5)
    hist_urgency = [
        _extract_urgency_from_historical(h)
        for h in historical_matches
    ]
    avg_hist_urgency = sum(hist_urgency) / len(hist_urgency) if hist_urgency else 0.5
    
    urgency_delta_pct = (
        ((current_urgency - avg_hist_urgency) / max(avg_hist_urgency, 0.1)) * 100
        if avg_hist_urgency > 0 else 0.0
    )
    
    current_text = _extract_text_from_analysis(current_analysis)
    current_keywords = _extract_keywords(current_text, top_n=20)
    
    hist_keywords_sets = [
        _extract_keywords(h.get("summary", ""), top_n=20)
        for h in historical_matches
    ]
    
    if hist_keywords_sets:
        overlaps = [
            _compute_jaccard(current_keywords, hist_kw)
            for hist_kw in hist_keywords_sets
        ]
        topic_overlap_pct = (sum(overlaps) / len(overlaps)) * 100
    else:
        topic_overlap_pct = 0.0
    
    return {
        "severity_delta_pct": round(severity_delta_pct, 2),
        "urgency_delta_pct": round(urgency_delta_pct, 2),
        "topic_overlap_pct": round(topic_overlap_pct, 2),
    }


def _severity_to_score(severity: str) -> float:
    """Map severity label to numeric score."""
    mapping = {
        "info": 1.0,
        "warning": 2.0,
        "critical": 3.0,
    }
    return mapping.get(severity.lower(), 1.0)


def _extract_severity_from_analysis(analysis: Dict[str, Any]) -> str:
    """Extract severity from current analysis."""
    return analysis.get("severity", "info")


def _extract_urgency_from_historical(historical: Dict[str, Any]) -> float:
    """Extract urgency score from historical match (fallback to severity)."""
    if "urgency_score" in historical:
        return historical["urgency_score"]
    
    severity = historical.get("severity", "info")
    return _severity_to_score(severity) / 3.0


def _extract_text_from_analysis(analysis: Dict[str, Any]) -> str:
    """Extract text content from analysis dict."""
    if "full_text" in analysis:
        return str(analysis["full_text"])
    
    if "conclusion" in analysis:
        return str(analysis["conclusion"])
    
    return str(analysis)[:500]


def _extract_keywords(text: str, top_n: int = 20) -> Set[str]:
    """Extract top keywords from text for topic overlap.
    
    Simple TF approach: lowercase words, remove common stopwords, count frequency.
    """
    if not text:
        return set()
    
    stopwords = {
        "the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
        "in", "with", "to", "for", "of", "as", "by", "from", "this", "that",
        "it", "was", "were", "are", "be", "been", "has", "have", "had",
    }
    
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    
    word_counts: Dict[str, int] = {}
    for word in words:
        if word not in stopwords:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {word for word, _ in sorted_words[:top_n]}


def _compute_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Jaccard similarity coefficient."""
    if not set1 and not set2:
        return 1.0
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0
