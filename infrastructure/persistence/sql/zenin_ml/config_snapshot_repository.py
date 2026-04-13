"""Repository for configuration snapshots (versioning)."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from iot_machine_learning.infrastructure.persistence.sql.zenin_db_connection import (
    ZeninDbConnection,
)

logger = logging.getLogger(__name__)


class ConfigSnapshotRepository:
    """Persists configuration snapshots for versioning."""

    def __init__(self, engine: Optional[Engine] = None):
        self._engine = engine or ZeninDbConnection.get_engine()

    def save_snapshot(
        self,
        config_type: str,
        config_data: Dict[str, Any],
        changed_by: Optional[str] = None,
        change_reason: Optional[str] = None,
    ) -> UUID:
        """Save a configuration snapshot.

        Args:
            config_type: Type of config ('feature_flags', 'engine_overrides', etc.)
            config_data: Configuration data as dict
            changed_by: Who made the change
            change_reason: Reason for change

        Returns:
            UUID of saved snapshot
        """
        try:
            # Compute hash
            config_json = json.dumps(config_data, sort_keys=True)
            config_hash = hashlib.sha256(config_json.encode()).hexdigest()

            # Check if identical config already exists
            if self._is_duplicate(config_type, config_hash):
                logger.debug(
                    "config_snapshot_duplicate",
                    extra={"config_type": config_type, "config_hash": config_hash},
                )
                return self._get_latest_id(config_type)  # type: ignore

            # Insert new snapshot
            snapshot_id = uuid4()

            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO zenin_ml.config_snapshots (
                            Id, ConfigType, ConfigData, ConfigHash,
                            ChangedBy, ChangeReason
                        ) VALUES (
                            :id, :config_type, :config_data, :config_hash,
                            :changed_by, :change_reason
                        )
                    """),
                    {
                        "id": str(snapshot_id),
                        "config_type": config_type,
                        "config_data": config_json,
                        "config_hash": config_hash,
                        "changed_by": changed_by,
                        "change_reason": change_reason,
                    },
                )

            logger.info(
                "config_snapshot_saved",
                extra={
                    "snapshot_id": str(snapshot_id),
                    "config_type": config_type,
                    "config_hash": config_hash,
                    "changed_by": changed_by,
                },
            )

            return snapshot_id

        except Exception as exc:
            logger.error(
                "config_snapshot_save_failed",
                extra={
                    "config_type": config_type,
                    "error": str(exc),
                },
            )
            raise

    def load_latest(
        self,
        config_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Load the latest configuration snapshot.

        Args:
            config_type: Type of config

        Returns:
            Configuration dict or None
        """
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text("""
                        SELECT TOP 1 ConfigData, SnapshotAt
                        FROM zenin_ml.config_snapshots
                        WHERE ConfigType = :config_type
                        ORDER BY SnapshotAt DESC
                    """),
                    {"config_type": config_type},
                ).fetchone()

            if result is None:
                return None

            return json.loads(result[0])

        except Exception as exc:
            logger.error(
                "config_snapshot_load_failed",
                extra={
                    "config_type": config_type,
                    "error": str(exc),
                },
            )
            return None

    def _is_duplicate(self, config_type: str, config_hash: str) -> bool:
        """Check if identical config already exists."""
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text("""
                        SELECT TOP 1 Id
                        FROM zenin_ml.config_snapshots
                        WHERE ConfigType = :config_type
                          AND ConfigHash = :config_hash
                        ORDER BY SnapshotAt DESC
                    """),
                    {"config_type": config_type, "config_hash": config_hash},
                ).fetchone()

            return result is not None

        except Exception:
            return False

    def _get_latest_id(self, config_type: str) -> Optional[UUID]:
        """Get ID of latest snapshot."""
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text("""
                        SELECT TOP 1 Id
                        FROM zenin_ml.config_snapshots
                        WHERE ConfigType = :config_type
                        ORDER BY SnapshotAt DESC
                    """),
                    {"config_type": config_type},
                ).fetchone()

            return UUID(result[0]) if result else None

        except Exception:
            return None
