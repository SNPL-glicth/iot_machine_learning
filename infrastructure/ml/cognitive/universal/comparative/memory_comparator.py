"""Fetch similar past analyses from CognitiveMemoryPort."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def fetch_similar_from_memory(
    cognitive_memory: object,
    query: str,
    series_id: Optional[str],
    domain: str,
    limit: int = 3,
    min_certainty: float = 0.7,
) -> List[Dict[str, Any]]:
    """Query CognitiveMemoryPort for similar past analyses.

    Args:
        cognitive_memory: CognitiveMemoryPort implementation
        query: Natural language query (e.g., full_text or summary)
        series_id: Optional series filter
        domain: Domain filter (via metadata)
        limit: Max results
        min_certainty: Minimum similarity threshold

    Returns:
        List of dicts with parsed memory results (empty on failure)
    """
    if not cognitive_memory:
        return []
    
    try:
        if hasattr(cognitive_memory, 'recall_similar_explanations'):
            results = cognitive_memory.recall_similar_explanations(
                query=query,
                series_id=series_id,
                limit=limit,
                min_certainty=min_certainty,
            )
            
            if results:
                return [_parse_memory_result(r) for r in results]
        
        if hasattr(cognitive_memory, 'recall_similar_anomalies'):
            results = cognitive_memory.recall_similar_anomalies(
                query=query,
                series_id=series_id,
                limit=limit,
                min_certainty=min_certainty,
            )
            
            if results:
                return [_parse_memory_result(r) for r in results]
    
    except Exception as e:
        logger.debug(f"memory_fetch_failed: {e}")
    
    return []


def _parse_memory_result(result) -> Dict[str, Any]:
    """Extract relevant fields from MemorySearchResult."""
    parsed = {
        "doc_id": getattr(result, 'object_id', 'unknown'),
        "score": getattr(result, 'score', 0.0),
        "summary": _extract_summary_from_memory(result),
        "severity": _extract_severity_from_memory(result),
        "timestamp": getattr(result, 'created_at', None),
        "resolution_time": None,
    }
    
    metadata = getattr(result, 'metadata', {})
    if isinstance(metadata, dict):
        parsed["resolution_time"] = metadata.get("resolution_time_hours")
    
    return parsed


def _extract_summary_from_memory(result) -> str:
    """Extract human-readable summary from MemorySearchResult."""
    content = getattr(result, 'content', '')
    
    if isinstance(content, str):
        lines = content.split('\n')
        for line in lines:
            if line.strip() and len(line) > 20:
                return line.strip()[:200]
    
    return str(content)[:200] if content else "No summary available"


def _extract_severity_from_memory(result) -> str:
    """Extract severity from MemorySearchResult metadata."""
    metadata = getattr(result, 'metadata', {})
    
    if isinstance(metadata, dict):
        return metadata.get('severity', 'info')
    
    return 'info'
