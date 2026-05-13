"""Series ID value object with validation (SEC-CRIT-3).

Prevents SQL injection, path traversal, and DoS attacks via malformed series IDs.

Applies OCP: Extensible to other ID types (TenantId, DeviceId) without modification.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, StringConstraints


# Validation constants (no magic numbers)
_SERIES_ID_MIN_LENGTH: int = 1
_SERIES_ID_MAX_LENGTH: int = 100
_SERIES_ID_PATTERN: str = r'^[a-zA-Z0-9_-]+$'


def _validate_series_id_format(value: str) -> str:
    """Validate series ID format (SEC-CRIT-3).
    
    Args:
        value: Series ID to validate.
    
    Returns:
        Validated series ID.
    
    Raises:
        ValueError: If series ID format is invalid.
    
    Applies SRP: Only validates format, no other concerns.
    """
    if not re.match(_SERIES_ID_PATTERN, value):
        raise ValueError(
            f"Invalid series_id format: '{value}'. "
            f"Must match pattern {_SERIES_ID_PATTERN} (alphanumeric, underscore, hyphen only)"
        )
    return value


# Type alias with validation (Pydantic v2 style)
SeriesId = Annotated[
    str,
    StringConstraints(
        min_length=_SERIES_ID_MIN_LENGTH,
        max_length=_SERIES_ID_MAX_LENGTH,
        strip_whitespace=True,
    ),
    AfterValidator(_validate_series_id_format),
]
"""Validated series ID type.

Constraints:
- Length: 1-100 characters
- Pattern: alphanumeric, underscore, hyphen only (no special chars)
- Whitespace: stripped automatically

Prevents:
- SQL injection (no quotes, semicolons)
- Path traversal (no slashes, dots)
- DoS (length limit)

Applies OCP: Can extend to TenantId, DeviceId with same pattern.
"""


# Optional: Specific ID types (extensible without modifying SeriesId)
TenantId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=50,
        strip_whitespace=True,
        pattern=r'^[a-zA-Z0-9_-]+$',
    ),
]
"""Validated tenant ID type (OCP: extends SeriesId pattern)."""


DeviceId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=50,
        strip_whitespace=True,
        pattern=r'^[a-zA-Z0-9_-]+$',
    ),
]
"""Validated device ID type (OCP: extends SeriesId pattern)."""


DocumentId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=100,
        strip_whitespace=True,
        pattern=r'^[a-zA-Z0-9_-]+$',
    ),
]
"""Validated document ID type (OCP: extends SeriesId pattern)."""
