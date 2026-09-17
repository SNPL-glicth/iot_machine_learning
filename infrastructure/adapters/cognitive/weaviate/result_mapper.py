"""Result mapping utilities for Weaviate query results."""

from __future__ import annotations

from typing import Any, Dict, Optional

from iot_machine_learning.domain.entities.memory_search_result import MemorySearchResult


def to_memory_result(
    raw: Dict[str, Any],
    text_field: str,
    *,
    series_key: Optional[str] = "seriesId",
) -> MemorySearchResult:
    """Convert a Weaviate GraphQL result dict to MemorySearchResult.
    
    Args:
        raw: Raw result dict from Weaviate GraphQL query
        text_field: Name of the field containing the main text content
        series_key: Name of the field containing series_id (None to skip)
        
    Returns:
        MemorySearchResult domain entity
    """
    additional = raw.get("_additional", {})
    source_id = raw.get("sourceRecordId")

    metadata = {
        k: v for k, v in raw.items()
        if k not in ("_additional", text_field, "seriesId",
                     "sourceRecordId", "createdAt")
        and v is not None
    }

    series_id = ""
    if series_key and series_key in raw:
        series_id = str(raw[series_key])
    elif "affectedSeriesIds" in raw:
        ids = raw["affectedSeriesIds"]
        series_id = ids[0] if ids else ""

    return MemorySearchResult(
        memory_id=additional.get("id", ""),
        series_id=series_id,
        text=raw.get(text_field, ""),
        certainty=float(additional.get("certainty", 0.0)),
        source_record_id=int(source_id) if source_id else None,
        created_at=raw.get("createdAt"),
        metadata=metadata,
    )
