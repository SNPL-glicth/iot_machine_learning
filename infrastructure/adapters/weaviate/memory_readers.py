"""Memory read operations for Weaviate cognitive memory."""

from __future__ import annotations

import logging
from typing import List, Optional

from iot_machine_learning.domain.entities.memory_search_result import MemorySearchResult
from .filter_builders import build_where_filter, where_eq_int, where_eq_text
from .query_operations import graphql_near_text
from .result_mapper import to_memory_result

logger = logging.getLogger(__name__)


def recall_similar_explanations(
    graphql_url: str,
    query: str,
    *,
    series_id: Optional[str] = None,
    engine_name: Optional[str] = None,
    limit: int = 5,
    min_certainty: float = 0.7,
    enabled: bool = True,
    dry_run: bool = False,
    timeout: int = 10,
) -> List[MemorySearchResult]:
    """Recall similar prediction explanations from memory.
    
    Args:
        graphql_url: Full URL to /v1/graphql endpoint
        query: Semantic search query
        series_id: Filter by series ID (optional)
        engine_name: Filter by engine name (optional)
        limit: Max results to return
        min_certainty: Minimum certainty threshold
        enabled: Master switch
        dry_run: If True, logs but doesn't send
        timeout: Request timeout
        
    Returns:
        List of MemorySearchResult objects
    """
    try:
        where = build_where_filter([
            where_eq_text("seriesId", series_id),
            where_eq_text("engineName", engine_name),
        ])
        results = graphql_near_text(
            graphql_url,
            "MLExplanation",
            [query],
            ["seriesId", "explanationText", "confidenceScore",
             "sourceRecordId", "createdAt", "engineName", "trend"],
            where_filter=where,
            limit=limit,
            certainty=min_certainty,
            enabled=enabled,
            dry_run=dry_run,
            timeout=timeout,
        )
        return [to_memory_result(r, "explanationText") for r in results]
    except Exception as exc:
        logger.warning(
            "recall_explanations_error",
            extra={"error": str(exc)},
        )
        return []


def recall_similar_anomalies(
    graphql_url: str,
    query: str,
    *,
    series_id: Optional[str] = None,
    severity: Optional[str] = None,
    event_code: Optional[str] = None,
    limit: int = 5,
    min_certainty: float = 0.7,
    enabled: bool = True,
    dry_run: bool = False,
    timeout: int = 10,
) -> List[MemorySearchResult]:
    """Recall similar anomaly detections from memory.
    
    Args:
        graphql_url: Full URL to /v1/graphql endpoint
        query: Semantic search query
        series_id: Filter by series ID (optional)
        severity: Filter by severity (optional)
        event_code: Filter by event code (optional)
        limit: Max results to return
        min_certainty: Minimum certainty threshold
        enabled: Master switch
        dry_run: If True, logs but doesn't send
        timeout: Request timeout
        
    Returns:
        List of MemorySearchResult objects
    """
    try:
        where = build_where_filter([
            where_eq_text("seriesId", series_id),
            where_eq_text("severity", severity),
            where_eq_text("eventCode", event_code),
        ])
        results = graphql_near_text(
            graphql_url,
            "AnomalyMemory",
            [query],
            ["seriesId", "explanationText", "anomalyScore", "severity",
             "sourceRecordId", "createdAt", "eventCode", "behaviorPattern"],
            where_filter=where,
            limit=limit,
            certainty=min_certainty,
            enabled=enabled,
            dry_run=dry_run,
            timeout=timeout,
        )
        return [to_memory_result(r, "explanationText") for r in results]
    except Exception as exc:
        logger.warning(
            "recall_anomalies_error",
            extra={"error": str(exc)},
        )
        return []


def recall_similar_patterns(
    graphql_url: str,
    query: str,
    *,
    series_id: Optional[str] = None,
    pattern_type: Optional[str] = None,
    limit: int = 5,
    min_certainty: float = 0.7,
    enabled: bool = True,
    dry_run: bool = False,
    timeout: int = 10,
) -> List[MemorySearchResult]:
    """Recall similar pattern detections from memory.
    
    Args:
        graphql_url: Full URL to /v1/graphql endpoint
        query: Semantic search query
        series_id: Filter by series ID (optional)
        pattern_type: Filter by pattern type (optional)
        limit: Max results to return
        min_certainty: Minimum certainty threshold
        enabled: Master switch
        dry_run: If True, logs but doesn't send
        timeout: Request timeout
        
    Returns:
        List of MemorySearchResult objects
    """
    try:
        where = build_where_filter([
            where_eq_text("seriesId", series_id),
            where_eq_text("patternType", pattern_type),
        ])
        results = graphql_near_text(
            graphql_url,
            "PatternMemory",
            [query],
            ["seriesId", "descriptionText", "patternType", "confidence",
             "sourceRecordId", "createdAt"],
            where_filter=where,
            limit=limit,
            certainty=min_certainty,
            enabled=enabled,
            dry_run=dry_run,
            timeout=timeout,
        )
        return [to_memory_result(r, "descriptionText") for r in results]
    except Exception as exc:
        logger.warning(
            "recall_patterns_error",
            extra={"error": str(exc)},
        )
        return []


def recall_similar_decisions(
    graphql_url: str,
    query: str,
    *,
    device_id: Optional[int] = None,
    severity: Optional[str] = None,
    limit: int = 5,
    min_certainty: float = 0.7,
    enabled: bool = True,
    dry_run: bool = False,
    timeout: int = 10,
) -> List[MemorySearchResult]:
    """Recall similar decision reasoning from memory.
    
    Args:
        graphql_url: Full URL to /v1/graphql endpoint
        query: Semantic search query
        device_id: Filter by device ID (optional)
        severity: Filter by severity (optional)
        limit: Max results to return
        min_certainty: Minimum certainty threshold
        enabled: Master switch
        dry_run: If True, logs but doesn't send
        timeout: Request timeout
        
    Returns:
        List of MemorySearchResult objects
    """
    try:
        where = build_where_filter([
            where_eq_int("deviceId", device_id),
            where_eq_text("severity", severity),
        ])
        results = graphql_near_text(
            graphql_url,
            "DecisionReasoning",
            [query],
            ["summaryText", "explanationText", "severity", "decisionType",
             "sourceRecordId", "createdAt", "affectedSeriesIds"],
            where_filter=where,
            limit=limit,
            certainty=min_certainty,
            enabled=enabled,
            dry_run=dry_run,
            timeout=timeout,
        )
        return [
            to_memory_result(r, "summaryText", series_key=None)
            for r in results
        ]
    except Exception as exc:
        logger.warning(
            "recall_decisions_error",
            extra={"error": str(exc)},
        )
        return []
