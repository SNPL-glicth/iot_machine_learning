"""Auto-detect input type from raw data."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

from .types import InputType


def detect_input_type(raw_data: Any) -> Tuple[InputType, Dict[str, Any]]:
    """Auto-detect input type and extract basic metadata.

    Args:
        raw_data: Any input (str, list, dict, etc.)

    Returns:
        Tuple of (InputType, metadata_dict)
    """
    if isinstance(raw_data, str):
        return _detect_text_or_special_chars(raw_data)
    
    if isinstance(raw_data, (list, tuple)):
        return _detect_numeric_or_mixed(raw_data)
    
    if isinstance(raw_data, dict):
        return _detect_tabular_or_json(raw_data)
    
    return InputType.UNKNOWN, {"error": "unrecognized_type"}


def _detect_text_or_special_chars(text: str) -> Tuple[InputType, Dict[str, Any]]:
    """Detect TEXT vs SPECIAL_CHARS."""
    if not text.strip():
        return InputType.UNKNOWN, {"error": "empty_string"}
    
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    
    alnum_ratio = sum(1 for c in text if c.isalnum()) / max(char_count, 1)
    
    if alnum_ratio < 0.4:
        return InputType.SPECIAL_CHARS, {
            "word_count": word_count,
            "char_count": char_count,
            "alnum_ratio": round(alnum_ratio, 3),
        }
    
    if word_count < 10:
        return InputType.UNKNOWN, {
            "word_count": word_count,
            "char_count": char_count,
            "reason": "too_short",
        }
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    return InputType.TEXT, {
        "word_count": word_count,
        "char_count": char_count,
        "paragraph_count": len(paragraphs),
        "alnum_ratio": round(alnum_ratio, 3),
    }


def _detect_numeric_or_mixed(values: list) -> Tuple[InputType, Dict[str, Any]]:
    """Detect NUMERIC vs MIXED list."""
    if not values:
        return InputType.UNKNOWN, {"error": "empty_list"}
    
    numeric_count = 0
    for v in values:
        try:
            if isinstance(v, (int, float)) and math.isfinite(v):
                numeric_count += 1
        except (TypeError, ValueError):
            pass
    
    if numeric_count == len(values):
        return InputType.NUMERIC, {
            "n_points": len(values),
            "has_timestamps": False,
        }
    
    if numeric_count > 0:
        return InputType.MIXED, {
            "n_points": len(values),
            "numeric_count": numeric_count,
            "non_numeric_count": len(values) - numeric_count,
        }
    
    return InputType.UNKNOWN, {"error": "no_numeric_values"}


def _detect_tabular_or_json(data: dict) -> Tuple[InputType, Dict[str, Any]]:
    """Detect TABULAR vs JSON."""
    if not data:
        return InputType.UNKNOWN, {"error": "empty_dict"}
    
    values = list(data.values())
    
    if not values:
        return InputType.JSON, {"n_keys": len(data)}
    
    all_lists = all(isinstance(v, list) for v in values)
    
    if all_lists:
        lengths = [len(v) for v in values]
        if len(set(lengths)) == 1 and lengths[0] > 0:
            numeric_columns = []
            for col_name, col_vals in data.items():
                if all(isinstance(v, (int, float)) and math.isfinite(v) for v in col_vals):
                    numeric_columns.append(col_name)
            
            return InputType.TABULAR, {
                "n_rows": lengths[0],
                "n_columns": len(data),
                "column_names": list(data.keys()),
                "numeric_columns": numeric_columns,
            }
    
    return InputType.JSON, {
        "n_keys": len(data),
        "has_nested": any(isinstance(v, (dict, list)) for v in values),
    }
