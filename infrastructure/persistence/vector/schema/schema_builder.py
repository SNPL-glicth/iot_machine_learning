"""Schema builder for Weaviate cognitive memory."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from .class_definitions import (
    ml_explanation_class,
    anomaly_memory_class,
    pattern_memory_class,
    decision_reasoning_class,
)

logger = logging.getLogger(__name__)


def get_all_classes() -> List[Dict[str, Any]]:
    """Return all 4 cognitive memory class definitions."""
    return [
        ml_explanation_class(),
        anomaly_memory_class(),
        pattern_memory_class(),
        decision_reasoning_class(),
    ]


def create_schema(
    weaviate_url: str,
    *,
    dry_run: bool = False,
    recreate: bool = False,
) -> bool:
    """Create the cognitive memory schema in Weaviate.

    Args:
        weaviate_url: Weaviate REST API URL.
        dry_run: If ``True``, print the schema JSON and exit.
        recreate: If ``True``, delete existing classes before creating.

    Returns:
        ``True`` if all classes were created successfully.
    """
    classes = get_all_classes()

    if dry_run:
        schema = {"classes": classes}
        print(json.dumps(schema, indent=2, ensure_ascii=False))
        logger.info("Dry-run: printed schema JSON for %d classes.", len(classes))
        return True

    try:
        import weaviate
    except ImportError:
        logger.error(
            "weaviate-client not installed. "
            "Run: pip install weaviate-client>=4.0.0"
        )
        return False

    logger.info("Connecting to Weaviate at %s ...", weaviate_url)

    try:
        client = weaviate.connect_to_local(
            host=weaviate_url.replace("http://", "").split(":")[0],
            port=int(weaviate_url.split(":")[-1]),
        )
    except Exception:
        logger.exception("Failed to connect to Weaviate at %s", weaviate_url)
        return False

    try:
        if not client.is_ready():
            logger.error("Weaviate is not ready at %s", weaviate_url)
            return False

        logger.info("Weaviate is ready. Creating schema...")

        existing = {c.name for c in client.collections.list_all().values()}

        # Import here to avoid circular dependency
        from .migration_runner import create_class_v4

        for class_def in classes:
            class_name = class_def["class"]

            if class_name in existing:
                if recreate:
                    logger.warning(
                        "Deleting existing class '%s' (--recreate).", class_name
                    )
                    client.collections.delete(class_name)
                else:
                    logger.info(
                        "Class '%s' already exists. Skipping.", class_name
                    )
                    continue

            create_class_v4(client, class_def)
            logger.info("Created class '%s' (%d properties).",
                        class_name, len(class_def["properties"]))

        logger.info("Schema creation complete.")
        return True

    finally:
        client.close()
