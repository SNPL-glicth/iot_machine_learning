"""Statistics and data resolution for numeric columns.

Utility functions for data processing and validation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def resolve_series(
    raw_series: Dict[str, List[float]],
    numeric_columns: List[str],
    sample_rows: List[Dict],
) -> Dict[str, List[float]]:
    """Resolve numeric column values from raw_series or sample_rows.

    Args:
        raw_series: Direct column→values mapping from parser.
        numeric_columns: Names of numeric columns.
        sample_rows: Fallback: list of row dicts.

    Returns:
        Dict of column_name → list of finite float values.
    """
    result: Dict[str, List[float]] = {}

    # Prefer raw_series (full data from .NET parser)
    if raw_series:
        for col_name, vals in raw_series.items():
            clean = [float(v) for v in vals if _is_finite(v)]
            if clean:
                result[col_name] = clean
        return result

    # Fallback: extract from sample_rows
    for col in numeric_columns:
        vals = []
        for row in sample_rows:
            raw = row.get(col)
            if raw is not None:
                try:
                    v = float(raw)
                    if math.isfinite(v):
                        vals.append(v)
                except (ValueError, TypeError):
                    continue
        if vals:
            result[col] = vals

    return result


def _is_finite(v) -> bool:
    """Check if a value is a finite number."""
    try:
        return math.isfinite(float(v))
    except (ValueError, TypeError):
        return False
