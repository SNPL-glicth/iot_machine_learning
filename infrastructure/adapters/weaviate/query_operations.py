"""GraphQL query operations for Weaviate semantic search."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .http_client import post_json

logger = logging.getLogger(__name__)


def graphql_near_text(
    graphql_url: str,
    class_name: str,
    concepts: List[str],
    return_fields: List[str],
    *,
    where_filter: Optional[Dict[str, Any]] = None,
    limit: int = 5,
    certainty: float = 0.7,
    enabled: bool = True,
    dry_run: bool = False,
    timeout: int = 10,
) -> List[Dict[str, Any]]:
    """Execute a nearText GraphQL query. Returns list of result dicts.
    
    Args:
        graphql_url: Full URL to /v1/graphql endpoint
        class_name: Weaviate class name
        concepts: List of search concepts
        return_fields: List of field names to return
        where_filter: Optional where filter dict
        limit: Max results to return
        certainty: Minimum certainty threshold (0.0-1.0)
        enabled: Master switch (if False, returns [])
        dry_run: If True, logs query but doesn't send
        timeout: Request timeout in seconds
        
    Returns:
        List of result dicts with requested fields + _additional metadata
    """
    if not enabled:
        return []

    fields_str = " ".join(return_fields)
    additional = "_additional { id certainty }"

    concepts_json = json.dumps(concepts)
    near_text = f'nearText: {{ concepts: {concepts_json}, certainty: {certainty} }}'

    where_clause = ""
    if where_filter:
        where_json = json.dumps(where_filter, default=str)
        where_clause = f", where: {where_json}"

    query = (
        "{ Get { "
        f"{class_name}({near_text}{where_clause}, limit: {limit}) "
        f"{{ {fields_str} {additional} }} "
        "} }"
    )

    if dry_run:
        logger.info(
            "weaviate_dry_run_query",
            extra={"class": class_name, "query": query},
        )
        return []

    resp = post_json(graphql_url, {"query": query}, timeout=timeout)
    if not resp:
        return []

    try:
        results = resp["data"]["Get"][class_name]
        return results if results else []
    except (KeyError, TypeError):
        errors = resp.get("errors", [])
        if errors:
            logger.warning(
                "weaviate_graphql_errors",
                extra={"class": class_name, "errors": errors[:3]},
            )
        return []
