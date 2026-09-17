"""Filter builders and utility functions for Weaviate queries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def build_where_filter(
    operands: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build a Weaviate where filter from a list of operand dicts.
    
    Skips None-valued operands. Returns None if no operands remain.
    
    Args:
        operands: List of filter operand dicts (None values are skipped)
        
    Returns:
        Combined filter dict, or None if no valid operands
    """
    valid = [op for op in operands if op is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    return {"operator": "And", "operands": valid}


def where_eq_text(path: str, value: Optional[str]) -> Optional[Dict[str, Any]]:
    """Build a text equality filter operand.
    
    Args:
        path: Property path
        value: Text value to match (if None, returns None)
        
    Returns:
        Filter operand dict, or None if value is None
    """
    if value is None:
        return None
    return {
        "path": [path],
        "operator": "Equal",
        "valueText": value,
    }


def where_eq_int(path: str, value: Optional[int]) -> Optional[Dict[str, Any]]:
    """Build an integer equality filter operand.
    
    Args:
        path: Property path
        value: Integer value to match (if None, returns None)
        
    Returns:
        Filter operand dict, or None if value is None
    """
    if value is None:
        return None
    return {
        "path": [path],
        "operator": "Equal",
        "valueInt": value,
    }


def now_iso() -> str:
    """Current UTC timestamp in ISO 8601 format.
    
    Returns:
        ISO 8601 timestamp string (e.g. "2026-02-18T13:57:00Z")
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_json(obj: Any) -> str:
    """Serialize to JSON string, handling non-serializable types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string, or "{}" on serialization error
    """
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return "{}"
