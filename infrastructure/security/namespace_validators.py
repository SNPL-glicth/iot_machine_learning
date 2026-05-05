"""Namespace ID validation for Redis keys."""
from __future__ import annotations
import logging
import re

logger = logging.getLogger(__name__)

_VALID_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
_MAX_ID_LENGTH = 64


def validate_and_sanitize(value: str, field_name: str, strict: bool) -> str:
    """Validate and optionally sanitize ID.

    Args:
        value: ID to validate
        field_name: Name of field (for error messages)
        strict: If True, raise on invalid; if False, sanitize

    Returns:
        Validated/sanitized ID

    Raises:
        ValueError: If invalid and strict=True
    """
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    if len(value) > _MAX_ID_LENGTH:
        raise ValueError(f"{field_name} too long: {len(value)} > {_MAX_ID_LENGTH}")
    if ':' in value:
        raise ValueError(f"{field_name} '{value}' contains ':' (key injection risk)")
    if not _VALID_ID_PATTERN.match(value):
        if strict:
            raise ValueError(
                f"{field_name} '{value}' contains invalid characters. "
                f"Only alphanumeric, dash, and underscore allowed."
            )
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', value)
        logger.warning(
            f"redis_namespace_sanitized: {field_name}='{value}' → '{sanitized}'"
        )
        return sanitized
    return value
