"""SQL Adapter for IoT Sensor Alert Thresholds.

Queries relational database tables (dbo.alert_thresholds) for user-defined
sensor boundaries using SQLAlchemy. Decoupled from ML Core.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


class SeveritySqlAdapter:
    """SQL database adapter for retrieving sensor alert thresholds."""

    def get_user_defined_range(
        self,
        conn: Connection,
        sensor_id: int,
    ) -> Optional[Tuple[float, float]]:
        """Queries the user-defined valid range from dbo.alert_thresholds."""
        row = conn.execute(
            text(
                """
                SELECT
                    threshold_value_min,
                    threshold_value_max
                FROM dbo.alert_thresholds
                WHERE sensor_id = :sensor_id
                  AND is_active = 1
                  AND condition_type = 'out_of_range'
                ORDER BY
                    CASE severity WHEN 'warning' THEN 0 ELSE 1 END,
                    id ASC
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchone()

        if not row:
            return None

        min_val = float(row[0]) if row[0] is not None else None
        max_val = float(row[1]) if row[1] is not None else None

        if min_val is None and max_val is None:
            return None

        if min_val is None:
            min_val = float("-inf")
        if max_val is None:
            max_val = float("inf")

        return (min_val, max_val)

    def is_value_within_user_thresholds(
        self,
        conn: Connection,
        sensor_id: int,
        value: float,
    ) -> bool:
        """Checks if a sensor value is within user-defined warning thresholds."""
        row = conn.execute(
            text(
                """
                SELECT
                    threshold_value_min,
                    threshold_value_max
                FROM dbo.alert_thresholds
                WHERE sensor_id = :sensor_id
                  AND is_active = 1
                  AND severity = 'warning'
                  AND condition_type = 'out_of_range'
                ORDER BY id ASC
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchone()

        if not row:
            return False

        warning_min = float(row[0]) if row[0] is not None else None
        warning_max = float(row[1]) if row[1] is not None else None

        if warning_min is None and warning_max is None:
            return False

        if warning_min is not None and value < warning_min:
            return False
        if warning_max is not None and value > warning_max:
            return False

        return True
