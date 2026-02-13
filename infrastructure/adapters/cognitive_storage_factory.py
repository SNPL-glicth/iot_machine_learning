"""Factory for creating a StoragePort with optional cognitive memory decoration.

Centralises the DI wiring decision:
    - If ``ML_ENABLE_COGNITIVE_MEMORY`` is ``True`` and a URL is configured,
      wraps the SQL adapter with ``CognitiveStorageDecorator``.
    - Otherwise, returns the SQL adapter as-is (zero overhead).

Usage::

    from iot_machine_learning.infrastructure.adapters.cognitive_storage_factory import (
        build_storage,
    )

    storage = build_storage(conn, flags)
    # ``storage`` is a ``StoragePort`` — callers don't know or care
    # whether cognitive memory is active.
"""

from __future__ import annotations

import logging

from sqlalchemy.engine import Connection

from ...domain.ports.storage_port import StoragePort
from ...ml_service.config.feature_flags import FeatureFlags
from .cognitive_storage_decorator import CognitiveStorageDecorator
from .null_cognitive import NullCognitiveAdapter
from .sqlserver_storage import SqlServerStorageAdapter
from .weaviate_cognitive import WeaviateCognitiveAdapter

logger = logging.getLogger(__name__)


def build_storage(
    conn: Connection,
    flags: FeatureFlags,
) -> StoragePort:
    """Build a ``StoragePort``, optionally decorated with cognitive memory.

    Args:
        conn: Active SQLAlchemy connection.
        flags: Feature flags (controls whether cognitive layer is active).

    Returns:
        A ``StoragePort`` instance.  If cognitive memory is enabled,
        this is a ``CognitiveStorageDecorator`` wrapping
        ``SqlServerStorageAdapter``.  Otherwise, it's the raw
        ``SqlServerStorageAdapter``.
    """
    sql_storage = SqlServerStorageAdapter(conn)

    if not flags.ML_ENABLE_COGNITIVE_MEMORY:
        logger.debug("cognitive_memory_disabled — using raw SQL storage")
        return sql_storage

    url = flags.ML_COGNITIVE_MEMORY_URL
    if not url:
        logger.warning(
            "cognitive_memory_enabled_but_no_url",
            extra={"info": "ML_COGNITIVE_MEMORY_URL is empty, falling back to NullCognitiveAdapter"},
        )
        cognitive = NullCognitiveAdapter()
    else:
        cognitive = WeaviateCognitiveAdapter(
            base_url=url,
            enabled=True,
            dry_run=flags.ML_COGNITIVE_MEMORY_DRY_RUN,
        )

    logger.info(
        "cognitive_storage_decorator_active",
        extra={
            "url": url,
            "dry_run": flags.ML_COGNITIVE_MEMORY_DRY_RUN,
            "async": flags.ML_COGNITIVE_MEMORY_ASYNC,
        },
    )

    return CognitiveStorageDecorator(sql_storage, cognitive, flags)
