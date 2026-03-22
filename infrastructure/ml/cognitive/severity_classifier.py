"""Severity classifier for IoT sensor predictions.

This module implements severity classification combining statistical anomalies with physical risk.
Part of the ZENIN ML cognitive pipeline for risk assessment and alerting.

REFACTORED: Delegates business rules to domain/services/severity_rules.py.

Before (3 mixed responsibilities):
  - Business rules: DEFAULT_RANGES, compute_risk_level, compute_severity
  - I/O: get_user_defined_range, is_value_within_user_thresholds (SQL)
  - Narrative: recommended_action generation

Now (1 responsibility):
  - I/O only (SQL queries) + delegation to domain rules

Extracted modules:
  - domain/entities/sensor_ranges.py (DEFAULT_RANGES as domain entity)
  - domain/services/severity_rules.py (pure rules: risk_level, severity, classify)

Security: No PII processed. Sensor values are validated and sanitized before classification.
All database queries use parameterized statements to prevent injection.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Connection

from iot_machine_learning.domain.services.severity_rules import (
    SeverityResult,
    classify_severity,
    compute_risk_level,
    compute_severity,
    is_out_of_range,
)
from iot_machine_learning.domain.entities.sensor_ranges import get_default_range

logger = logging.getLogger(__name__)

# Re-export para compatibilidad con imports existentes
__all__ = ["SeverityClassifier", "SeverityResult"]


class SeverityClassifier:
    """Clasifica severidad combinando anomalía estadística y riesgo físico.

    Responsabilidad: I/O (queries SQL) + delegación a domain rules.
    No contiene reglas de negocio — esas están en domain/services/severity_rules.py.
    """

    def get_user_defined_range(
        self,
        conn: Connection,
        sensor_id: int,
    ) -> Optional[Tuple[float, float]]:
        """Obtiene el rango definido por el usuario desde alert_thresholds.

        Responsabilidad: solo I/O (SQL query).
        """
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
            min_val = float('-inf')
        if max_val is None:
            max_val = float('inf')

        return (min_val, max_val)

    def is_value_within_user_thresholds(
        self,
        conn: Connection,
        sensor_id: int,
        value: float,
    ) -> bool:
        """Verifica si el valor está dentro de los umbrales WARNING del usuario.

        Responsabilidad: solo I/O (SQL query).
        """
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

    def classify(
        self,
        *,
        sensor_type: str,
        location: str,
        predicted_value: float,
        trend: str,
        anomaly: bool,
        anomaly_score: float,
        confidence: float,
        horizon_minutes: int,
        user_defined_range: Optional[Tuple[float, float]] = None,
    ) -> SeverityResult:
        """Clasifica severidad completa — delega a domain rules.

        Mantiene la misma API pública para compatibilidad.

        Returns:
            SeverityResult con risk_level, severity, action_required, recommended_action
        """
        return classify_severity(
            sensor_type=sensor_type,
            location=location,
            predicted_value=predicted_value,
            anomaly=anomaly,
            user_defined_range=user_defined_range,
        )
