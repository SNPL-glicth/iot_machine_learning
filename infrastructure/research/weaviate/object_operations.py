"""Object creation operations for Weaviate."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .http_client import post_json

logger = logging.getLogger(__name__)


def create_object(
    objects_url: str,
    class_name: str,
    properties: Dict[str, Any],
    *,
    enabled: bool = True,
    dry_run: bool = False,
    timeout: int = 10,
) -> Optional[str]:
    """Create a Weaviate object. Returns UUID or None.
    
    Args:
        objects_url: Full URL to /v1/objects endpoint
        class_name: Weaviate class name
        properties: Object properties dict
        enabled: Master switch (if False, returns None)
        dry_run: If True, logs payload but doesn't send
        timeout: Request timeout in seconds
        
    Returns:
        Object UUID string, or None on failure
    """
    if not enabled:
        return None

    payload = {"class": class_name, "properties": properties}

    if dry_run:
        logger.info(
            "weaviate_dry_run_create",
            extra={"class": class_name, "properties": properties},
        )
        return "dry-run-uuid"

    resp = post_json(objects_url, payload, timeout=timeout)
    if resp and "id" in resp:
        uuid = resp["id"]
        logger.debug(
            "weaviate_object_created",
            extra={"class": class_name, "uuid": uuid},
        )
        return uuid

    logger.warning(
        "weaviate_create_failed",
        extra={"class": class_name, "response": resp},
    )
    return None
