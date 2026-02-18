"""Tests for COG-4 — Unified severity thresholds.

Requires sqlalchemy (transitive dep of ml_service.explain).
Placed in integration/ since it depends on ml_service infrastructure.
"""

from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip(
    "sqlalchemy", reason="sqlalchemy required for ml_service.explain"
)


class TestUnifiedSeverityThresholds:

    def _make_context(self, **overrides):
        from iot_machine_learning.ml_service.explain.models.enriched_context import (
            EnrichedContext,
        )
        defaults = dict(
            sensor_id=1,
            sensor_name="temp_1",
            sensor_type="temperature",
            location="Room A",
            unit="°C",
            device_id=1,  # Added
            device_name="Device 1",  # Added
            device_type="IoT Sensor",  # Added
            current_value=22.0,
            predicted_value=23.0,
            trend="rising",
            confidence=0.8,
            horizon_minutes=30,
            is_anomaly=False,
            anomaly_score=0.0,
            similar_events_count=0,
            correlated_events=[],
            user_threshold_min=None,
            user_threshold_max=None,
            recent_avg=None,
            recent_min=None,  # Added
            recent_max=None,  # Added
            recent_std=None,
            last_similar_event_at=None,  # Added
        )
        defaults.update(overrides)
        return EnrichedContext(**defaults)

    def test_low_score_returns_low(self) -> None:
        from iot_machine_learning.ml_service.explain.services.template_generator import (
            TemplateExplanationGenerator,
        )

        gen = TemplateExplanationGenerator()
        ctx = self._make_context(anomaly_score=0.1, is_anomaly=False)
        sev = gen._determine_severity(ctx)
        assert sev == "LOW"

    def test_medium_score_anomaly_returns_medium(self) -> None:
        from iot_machine_learning.ml_service.explain.services.template_generator import (
            TemplateExplanationGenerator,
        )

        gen = TemplateExplanationGenerator()
        ctx = self._make_context(anomaly_score=0.6, is_anomaly=True)
        sev = gen._determine_severity(ctx)
        assert sev == "MEDIUM"

    def test_high_score_anomaly_returns_high(self) -> None:
        from iot_machine_learning.ml_service.explain.services.template_generator import (
            TemplateExplanationGenerator,
        )

        gen = TemplateExplanationGenerator()
        ctx = self._make_context(anomaly_score=0.85, is_anomaly=True)
        sev = gen._determine_severity(ctx)
        assert sev == "HIGH"

    def test_critical_score_anomaly_returns_high(self) -> None:
        from iot_machine_learning.ml_service.explain.services.template_generator import (
            TemplateExplanationGenerator,
        )

        gen = TemplateExplanationGenerator()
        ctx = self._make_context(anomaly_score=0.95, is_anomaly=True)
        sev = gen._determine_severity(ctx)
        assert sev == "HIGH"

    def test_out_of_range_returns_critical(self) -> None:
        from iot_machine_learning.ml_service.explain.services.template_generator import (
            TemplateExplanationGenerator,
        )

        gen = TemplateExplanationGenerator()
        ctx = self._make_context(
            predicted_value=50.0,
            user_threshold_max=30.0,
            anomaly_score=0.1,
        )
        sev = gen._determine_severity(ctx)
        assert sev == "CRITICAL"

    def test_not_anomaly_but_high_score_returns_medium(self) -> None:
        from iot_machine_learning.ml_service.explain.services.template_generator import (
            TemplateExplanationGenerator,
        )

        gen = TemplateExplanationGenerator()
        ctx = self._make_context(anomaly_score=0.75, is_anomaly=False)
        sev = gen._determine_severity(ctx)
        assert sev == "MEDIUM"

    def test_severity_aligns_with_domain(self) -> None:
        from iot_machine_learning.domain.entities.results.anomaly import (
            AnomalySeverity,
        )
        from iot_machine_learning.ml_service.explain.services.template_generator import (
            TemplateExplanationGenerator,
        )

        gen = TemplateExplanationGenerator()

        ctx = self._make_context(anomaly_score=0.2, is_anomaly=False)
        assert gen._determine_severity(ctx) == "LOW"
        assert AnomalySeverity.from_score(0.2) == AnomalySeverity.NONE

        ctx = self._make_context(anomaly_score=0.6, is_anomaly=False)
        assert gen._determine_severity(ctx) == "MEDIUM"
        assert AnomalySeverity.from_score(0.6) == AnomalySeverity.MEDIUM
