"""Weaviate schema initializer — ensures classes exist before writes.

On startup:
    1. GET /v1/schema/{ClassName} for each cognitive-memory class.
    2. If 404 → POST /v1/schema with full class definition.
    3. If class exists → compare properties list and POST any missing ones
       to /v1/schema/{ClassName}/properties.

This guarantees that the first ``remember_explanation`` or
``recall_similar_explanations`` call never hits a missing-schema error.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from iot_machine_learning.infrastructure.persistence.vector.schema.class_definitions import (
    anomaly_memory_class,
    decision_reasoning_class,
    ml_explanation_class,
    pattern_memory_class,
)
from iot_machine_learning.infrastructure.persistence.vector.schema.property_builder import (
    build_property,
)

from .http_client import get_json, post_json

logger = logging.getLogger(__name__)

# All cognitive-memory classes that the ML Service writes or queries.
_COGNITIVE_CLASSES = [
    ml_explanation_class,
    anomaly_memory_class,
    pattern_memory_class,
    decision_reasoning_class,
]


def _schema_url(base_url: str, class_name: str) -> str:
    return f"{base_url.rstrip('/')}/v1/schema/{class_name}"


def _property_url(base_url: str, class_name: str) -> str:
    return f"{base_url.rstrip('/')}/v1/schema/{class_name}/properties"


def _get_existing_property_names(existing_class: Dict[str, Any]) -> Set[str]:
    """Extract property names from a Weaviate class schema response."""
    props = existing_class.get("properties", []) or existing_class.get("class", {}).get("properties", [])
    if not isinstance(props, list):
        return set()
    return {p.get("name") for p in props if p.get("name")}


def _ensure_class_exists(
    base_url: str,
    class_def: Dict[str, Any],
    timeout: int = 10,
) -> None:
    """Create a class if absent, or add missing properties if present."""
    class_name = class_def["class"]
    url = _schema_url(base_url, class_name)

    existing = get_json(url, timeout=timeout)

    if existing is None:
        # Class does not exist → create it
        create_url = f"{base_url.rstrip('/')}/v1/schema"
        result = post_json(create_url, class_def, timeout=timeout)
        if result is not None:
            logger.info(
                "weaviate_schema_created",
                extra={"class": class_name},
            )
        else:
            logger.warning(
                "weaviate_schema_create_failed",
                extra={"class": class_name},
            )
        return

    # Class exists → forward-migrate any missing properties
    existing_names = _get_existing_property_names(existing)
    missing_props: List[Dict[str, Any]] = []

    for prop in class_def.get("properties", []):
        prop_name = prop.get("name")
        if prop_name and prop_name not in existing_names:
            missing_props.append(prop)

    if not missing_props:
        logger.debug(
            "weaviate_schema_up_to_date",
            extra={"class": class_name},
        )
        return

    for prop in missing_props:
        prop_url = _property_url(base_url, class_name)
        result = post_json(prop_url, prop, timeout=timeout)
        if result is not None:
            logger.info(
                "weaviate_schema_property_added",
                extra={"class": class_name, "property": prop["name"]},
            )
        else:
            logger.warning(
                "weaviate_schema_property_add_failed",
                extra={"class": class_name, "property": prop["name"]},
            )


def ensure_schema_exists(
    client_url: str,
    *,
    timeout: int = 10,
    enabled: bool = True,
) -> None:
    """Ensure all cognitive-memory Weaviate classes exist and are up-to-date.

    Args:
        client_url: Base URL of the Weaviate instance (e.g. ``http://weaviate:8080``).
        timeout: HTTP timeout per request in seconds.
        enabled: Master switch.  If ``False``, the function is a no-op.
    """
    if not enabled or not client_url:
        return

    for class_factory in _COGNITIVE_CLASSES:
        try:
            class_def = class_factory()
            _ensure_class_exists(client_url, class_def, timeout=timeout)
        except Exception as exc:
            logger.warning(
                "weaviate_schema_init_error",
                extra={"error": str(exc)},
            )
