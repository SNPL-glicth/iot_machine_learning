"""Tests para FileAuditLogger y NullAuditLogger.

Verifica:
- Escritura de eventos a archivo JSON Lines
- Hash de integridad presente
- Helpers (log_prediction, log_anomaly, log_config_change)
- NullAuditLogger no crashea
- Formato ISO 27001 (timestamp UTC, user_id, action, resource)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from iot_machine_learning.infrastructure.security.audit_logger import (
    FileAuditLogger,
    NullAuditLogger,
)


@pytest.fixture
def audit_file(tmp_path: Path) -> Path:
    """Archivo temporal para audit log."""
    return tmp_path / "audit.jsonl"


class TestFileAuditLogger:
    """Tests para FileAuditLogger."""

    def test_log_event_writes_json_line(self, audit_file: Path) -> None:
        """Cada evento debe escribirse como una línea JSON."""
        logger = FileAuditLogger(log_file=audit_file, include_hash=True)

        logger.log_event(
            event_type="prediction",
            action="predict",
            resource="sensor_1",
            details={"predicted_value": 22.5},
        )

        lines = audit_file.read_text().strip().split("\n")
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["event_type"] == "prediction"
        assert entry["action"] == "predict"
        assert entry["resource"] == "sensor_1"
        assert entry["user_id"] == "system"
        assert entry["result"] == "success"
        assert "timestamp" in entry
        assert "integrity_hash" in entry

    def test_log_event_timestamp_is_utc_iso(self, audit_file: Path) -> None:
        """Timestamp debe ser UTC ISO format (ISO 27001 A.12.4.4)."""
        logger = FileAuditLogger(log_file=audit_file)

        logger.log_event(
            event_type="test",
            action="test",
            resource="test",
            details={},
        )

        entry = json.loads(audit_file.read_text().strip())
        ts = entry["timestamp"]
        # Debe contener 'T' (ISO format) y '+00:00' o 'Z' (UTC)
        assert "T" in ts
        assert "+00:00" in ts or "Z" in ts

    def test_log_prediction_helper(self, audit_file: Path) -> None:
        """Helper log_prediction debe escribir evento correcto."""
        logger = FileAuditLogger(log_file=audit_file)

        logger.log_prediction(
            sensor_id=42,
            predicted_value=22.5,
            confidence=0.85,
            engine_name="taylor",
            trace_id="abc123",
        )

        entry = json.loads(audit_file.read_text().strip())
        assert entry["event_type"] == "prediction"
        assert entry["details"]["engine"] == "taylor"
        assert entry["details"]["trace_id"] == "abc123"

    def test_log_anomaly_helper(self, audit_file: Path) -> None:
        """Helper log_anomaly debe escribir evento correcto."""
        logger = FileAuditLogger(log_file=audit_file)

        logger.log_anomaly(
            sensor_id=1,
            value=50.0,
            score=0.9,
            explanation="Z-score alto",
        )

        entry = json.loads(audit_file.read_text().strip())
        assert entry["event_type"] == "anomaly_detection"
        assert entry["details"]["anomaly_score"] == 0.9

    def test_log_config_change_before_after(self, audit_file: Path) -> None:
        """Cambio de config debe incluir before/after state."""
        logger = FileAuditLogger(log_file=audit_file)

        logger.log_config_change(
            config_key="ML_TAYLOR_ORDER",
            old_value=2,
            new_value=3,
            user_id="admin_user",
        )

        entry = json.loads(audit_file.read_text().strip())
        assert entry["event_type"] == "config_change"
        assert entry["user_id"] == "admin_user"
        assert entry["before_state"]["value"] == 2
        assert entry["after_state"]["value"] == 3

    def test_integrity_hash_present(self, audit_file: Path) -> None:
        """Hash de integridad debe estar presente."""
        logger = FileAuditLogger(log_file=audit_file, include_hash=True)

        logger.log_event(
            event_type="test",
            action="test",
            resource="test",
            details={},
        )

        entry = json.loads(audit_file.read_text().strip())
        assert "integrity_hash" in entry
        assert len(entry["integrity_hash"]) == 16  # SHA-256 truncado

    def test_no_hash_when_disabled(self, audit_file: Path) -> None:
        """Sin hash cuando include_hash=False."""
        logger = FileAuditLogger(log_file=audit_file, include_hash=False)

        logger.log_event(
            event_type="test",
            action="test",
            resource="test",
            details={},
        )

        entry = json.loads(audit_file.read_text().strip())
        assert "integrity_hash" not in entry

    def test_multiple_events(self, audit_file: Path) -> None:
        """Múltiples eventos deben escribirse en líneas separadas."""
        logger = FileAuditLogger(log_file=audit_file)

        for i in range(5):
            logger.log_event(
                event_type="test",
                action=f"action_{i}",
                resource="test",
                details={"index": i},
            )

        lines = audit_file.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Debe crear directorio padre si no existe."""
        nested_path = tmp_path / "subdir" / "deep" / "audit.jsonl"
        logger = FileAuditLogger(log_file=nested_path)

        logger.log_event(
            event_type="test",
            action="test",
            resource="test",
            details={},
        )

        assert nested_path.exists()


class TestNullAuditLogger:
    """Tests para NullAuditLogger (no-op)."""

    def test_no_crash_on_all_methods(self) -> None:
        """NullAuditLogger no debe crashear en ningún método."""
        logger = NullAuditLogger()

        logger.log_event("test", "test", "test", {})
        logger.log_prediction(1, 22.5, 0.85, "taylor")
        logger.log_anomaly(1, 50.0, 0.9, "test")
        logger.log_config_change("key", "old", "new", "user")
        # No debe lanzar excepciones
